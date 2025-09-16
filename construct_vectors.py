import os
import re
import math
import json
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Visualization imports
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HF_ENV_TOKEN_KEYS = ["HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"]


# ========== CONFIG ==========
@dataclass
class BuildConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    bnb_compute_dtype: str = "float16"  # or "bfloat16" if supported
    seed: int = 0

    # Data
    hf_dataset: str = "hasnat79/ACCD"
    hf_config: str = "train"
    max_samples_harmful: int = 1000
    max_samples_benign: int = 1000
    min_completion_chars: int = 12

    # Hidden-state extraction
    avg_k_first_assistant_tokens: int = 1
    # Layer search range as fractions of depth (inclusive). Set to 0.0-1.0 to search all layers.
    search_start_frac: float = 0.6
    search_end_frac: float = 1.0

    # Tokenization
    max_seq_len: int = 2048

    # Visualization
    enable_visualization: bool = True
    viz_samples_per_class: int = 500
    viz_reduction_methods: List[str] = None

    # Output
    out_dir: str = "steering_vectors_llama31_8b"
    out_prefix: str = "refusal_lastlayer"

    def __post_init__(self):
        if self.viz_reduction_methods is None:
            self.viz_reduction_methods = ["pca_3d"]


CFG = BuildConfig()


# ========== UTILS ==========
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def refusal_regex():
    return re.compile(
        r"(i\s*(?:am|'m)\s*(?:sorry|afraid)|\b(?:can(?:not|'t)\s+(?:help|provide|comply|assist))\b|\b(?:i\s+cannot|i\s+can\'t)\b)",
        re.IGNORECASE,
    )


def pick_user_prompt(row: Dict[str, str]) -> str:
    inp = row.get("input")
    if isinstance(inp, str) and inp.strip():
        return inp
    adv = row.get("adversarial")
    if adv and isinstance(adv, str) and len(adv.strip()) > 0:
        return adv
    return row.get("vanilla", "")


def is_valid_completion(text: str, min_chars: int) -> bool:
    return isinstance(text, str) and (len(text.strip()) >= min_chars)


# ========== LOAD MODEL ==========
def load_model_and_tokenizer():
    compute_dtype = torch.float16 if CFG.bnb_compute_dtype.lower() == "float16" else torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CFG.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    for k in HF_ENV_TOKEN_KEYS:
        if os.getenv(k):
            break
    tok = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        quantization_config=bnb_config if CFG.load_in_4bit else None,
        device_map="auto",
    )
    model.config.output_hidden_states = True
    model.eval()
    return tok, model


# ========== CHAT FORMATTING & HIDDEN STATES ==========
def chat_format(tok: AutoTokenizer, user: str, assistant: str) -> str:
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def first_assistant_hidden_seq_for_layers(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    user_text: str,
    assistant_text: str,
    layer_indices_1_based: List[int],
    k_tokens: int,
) -> Dict[int, torch.Tensor]:
    text_full = chat_format(tok, user_text, assistant_text)
    enc_full = tok(
        text_full,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_seq_len,
    )
    enc_full = {k: v.to(model.device) for k, v in enc_full.items()}

    user_prefix = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    user_ids = tok(user_prefix, return_tensors="pt")["input_ids"][0]
    boundary = int(user_ids.shape[0])

    out = model(**enc_full, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: [embed, layer1, ..., layerN]
    seq_len = hs[1].shape[1]

    if boundary >= seq_len:
        return {}

    start = boundary
    K = max(1, k_tokens)
    end = min(seq_len, boundary + K)
    if (end - start) < K:
        return {}

    per_layer_vec = {}
    for L in layer_indices_1_based:
        idx = max(1, min(L, len(hs) - 1))
        h_seq = hs[idx][0, start:end, :].to(torch.float32).contiguous()  # (K, d)
        per_layer_vec[L] = h_seq
    return per_layer_vec


# ========== VISUALIZATION HELPERS ==========
class LatentSpaceVisualizer:
    def __init__(self, refusal_vectors: List[np.ndarray], nonref_vectors: List[np.ndarray], layer_id: int, out_dir: str):
        self.refusal_vectors = np.array(refusal_vectors)
        self.nonref_vectors = np.array(nonref_vectors)
        self.layer_id = layer_id
        self.out_dir = out_dir

        self.all_vectors = np.vstack([self.refusal_vectors, self.nonref_vectors])
        self.labels = np.array([1] * len(self.refusal_vectors) + [0] * len(self.nonref_vectors))

        self.scaler = StandardScaler()
        self.all_vectors_scaled = self.scaler.fit_transform(self.all_vectors)

    def plot_pca_3d(self):
        pca = PCA(n_components=3)
        vectors_pca = pca.fit_transform(self.all_vectors_scaled)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            vectors_pca[:, 0], vectors_pca[:, 1], vectors_pca[:, 2],
            c=self.labels, cmap='coolwarm', alpha=0.7, s=30
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA 3D - Layer {self.layer_id}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f'pca3d_layer_{self.layer_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_tsne(self, perplexity=30):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=CFG.seed)
        vectors_tsne = tsne.fit_transform(self.all_vectors_scaled)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        scatter = ax1.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], c=self.labels, cmap='coolwarm', alpha=0.6, s=30)
        ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
        ax1.set_title(f't-SNE - Layer {self.layer_id}')
        plt.colorbar(scatter, ax=ax1, label='Refusal (1) / Non-refusal (0)')
        refusal_tsne = vectors_tsne[self.labels == 1]
        nonref_tsne = vectors_tsne[self.labels == 0]
        ax2.hexbin(refusal_tsne[:, 0], refusal_tsne[:, 1], gridsize=20, alpha=0.5, cmap='Reds', label='Refusal')
        ax2.hexbin(nonref_tsne[:, 0], nonref_tsne[:, 1], gridsize=20, alpha=0.5, cmap='Blues', label='Non-refusal')
        ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2'); ax2.set_title(f't-SNE Density - Layer {self.layer_id}')
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, f'tsne_layer_{self.layer_id}.png'), dpi=150, bbox_inches='tight'); plt.close()

    def plot_distance_distributions(self):
        refusal_centroid = np.mean(self.refusal_vectors, axis=0)
        nonref_centroid = np.mean(self.nonref_vectors, axis=0)
        refusal_distances = np.linalg.norm(self.refusal_vectors - refusal_centroid, axis=1)
        nonref_distances = np.linalg.norm(self.nonref_vectors - nonref_centroid, axis=1)
        inter_cluster_dist = np.linalg.norm(refusal_centroid - nonref_centroid)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        refusal_cos_sim = [cosine_similarity(v, refusal_centroid) for v in self.refusal_vectors]
        nonref_cos_sim = [cosine_similarity(v, nonref_centroid) for v in self.nonref_vectors]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].hist(refusal_distances, bins=30, alpha=0.7, color='red', label='Refusal', density=True)
        axes[0, 0].hist(nonref_distances, bins=30, alpha=0.7, color='blue', label='Non-refusal', density=True)
        axes[0, 0].axvline(inter_cluster_dist, color='green', linestyle='--', label=f'Inter-cluster: {inter_cluster_dist:.2f}')
        axes[0, 0].set_xlabel('Distance to Centroid'); axes[0, 0].set_ylabel('Density'); axes[0, 0].set_title(f'Within-Cluster Distances - Layer {self.layer_id}')
        axes[0, 0].legend()
        axes[0, 1].hist(refusal_cos_sim, bins=30, alpha=0.7, color='red', label='Refusal', density=True)
        axes[0, 1].hist(nonref_cos_sim, bins=30, alpha=0.7, color='blue', label='Non-refusal', density=True)
        axes[0, 1].set_xlabel('Cosine Similarity to Centroid'); axes[0, 1].set_ylabel('Density'); axes[0, 1].set_title(f'Cosine Similarities - Layer {self.layer_id}')
        axes[0, 1].legend()
        axes[1, 0].boxplot([refusal_distances, nonref_distances], labels=['Refusal', 'Non-refusal'])
        axes[1, 0].set_ylabel('Distance to Centroid'); axes[1, 0].set_title('Distance Distribution Comparison'); axes[1, 0].grid(True, alpha=0.3)
        stats_text = f"""Cluster Statistics (Layer {self.layer_id}):\n\nRefusal Cluster:\n  Mean distance: {np.mean(refusal_distances):.3f}\n  Std distance: {np.std(refusal_distances):.3f}\n  Mean cosine sim: {np.mean(refusal_cos_sim):.3f}\n  \nNon-refusal Cluster:\n  Mean distance: {np.mean(nonref_distances):.3f}\n  Std distance: {np.std(nonref_distances):.3f}\n  Mean cosine sim: {np.mean(nonref_cos_sim):.3f}\n  \nInter-cluster distance: {inter_cluster_dist:.3f}\nCentroid cosine sim: {np.dot(refusal_centroid, nonref_centroid)/(np.linalg.norm(refusal_centroid)*np.linalg.norm(nonref_centroid)):.3f}"""
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, f'distances_layer_{self.layer_id}.png'), dpi=150, bbox_inches='tight'); plt.close()

    def create_all_visualizations(self):
        print(f"  Creating visualizations for layer {self.layer_id}...")
        if "pca_3d" in CFG.viz_reduction_methods:
            self.plot_pca_3d()
        self.plot_distance_distributions()


def compute_separation_metrics(all_layer_embeddings: Dict[int, Dict[str, List[np.ndarray]]]) -> Dict[int, Dict[str, float]]:
    layers = sorted(all_layer_embeddings.keys())
    separation_metrics: Dict[int, Dict[str, float]] = {}
    for layer in layers:
        refusal_vecs = np.array(all_layer_embeddings[layer]['refusal'])
        nonref_vecs = np.array(all_layer_embeddings[layer]['nonref'])
        if len(refusal_vecs) < 2 or len(nonref_vecs) < 2:
            continue
        ref_centroid = np.mean(refusal_vecs, axis=0)
        nonref_centroid = np.mean(nonref_vecs, axis=0)
        inter_dist = float(np.linalg.norm(ref_centroid - nonref_centroid))
        ref_intra = float(np.mean([np.linalg.norm(v - ref_centroid) for v in refusal_vecs]))
        nonref_intra = float(np.mean([np.linalg.norm(v - nonref_centroid) for v in nonref_vecs]))
        avg_intra = (ref_intra + nonref_intra) / 2.0 if (ref_intra + nonref_intra) > 0 else 0.0
        ratio = (inter_dist / avg_intra) if avg_intra > 0 else 0.0
        separation_metrics[layer] = {
            'inter_dist': inter_dist,
            'avg_intra': avg_intra,
            'separation_ratio': ratio,
        }
    return separation_metrics


def plot_cross_layer_metrics(separation_metrics: Dict[int, Dict[str, float]], out_dir: str):
    if not separation_metrics:
        return
    layers = sorted(separation_metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(layers, [separation_metrics[l]['inter_dist'] for l in layers], 'bo-')
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('Inter-cluster Distance'); axes[0].set_title('Cluster Separation Across Layers'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(layers, [separation_metrics[l]['avg_intra'] for l in layers], 'ro-')
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Average Intra-cluster Distance'); axes[1].set_title('Cluster Compactness Across Layers'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(layers, [separation_metrics[l]['separation_ratio'] for l in layers], 'go-')
    axes[2].set_xlabel('Layer'); axes[2].set_ylabel('Separation Ratio'); axes[2].set_title('Cluster Quality (Inter/Intra Distance)'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'cross_layer_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()
    with open(os.path.join(out_dir, 'separation_metrics.json'), 'w') as f:
        json.dump(separation_metrics, f, indent=2)


# ========== MAIN BUILD ==========
def main():
    os.makedirs(CFG.out_dir, exist_ok=True)
    set_seed(CFG.seed)

    print(f"[INFO] Loading model: {CFG.model_name} (4-bit={CFG.load_in_4bit})")
    tok, model = load_model_and_tokenizer()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = len(getattr(model, "model").layers)

    # Define layer candidates to evaluate (1-based inclusive)
    L_start = max(1, int(math.ceil(num_layers * CFG.search_start_frac)))
    L_end = min(num_layers, int(math.floor(num_layers * CFG.search_end_frac)))
    layer_candidates = list(range(L_start, L_end + 1))
    print(f"[INFO] Evaluating layers {L_start}-{L_end} out of {num_layers}")

    print(f"[INFO] Loading dataset: {CFG.hf_dataset} (streaming)")
    ds_stream = load_dataset(CFG.hf_dataset, streaming=True)
    split_keys = list(ds_stream.keys())
    split = CFG.hf_config if CFG.hf_config in split_keys else split_keys[0]
    print(f"[INFO] Using split: {split}")

    # Pre-count balance to sample equal numbers when labels exist
    temp_iter = iter(ds_stream[split])
    cnt_safe = 0
    cnt_unsafe = 0
    while True:
        try:
            row = next(temp_iter)
        except StopIteration:
            break
        assistant_text = row.get("target", "") or row.get("completion", "")
        if not is_valid_completion(assistant_text, CFG.min_completion_chars):
            continue
        sl = row.get("safety_label")
        if sl == "safe":
            cnt_safe += 1
        elif sl == "unsafe":
            cnt_unsafe += 1
    if cnt_safe > 0 or cnt_unsafe > 0:
        equal_target = min(cnt_safe, cnt_unsafe)
        if equal_target == 0:
            print(f"[WARN] One class has zero valid samples (safe={cnt_safe}, unsafe={cnt_unsafe}).")
        target_refusal = equal_target
        target_nonref = equal_target
    else:
        target_refusal = CFG.max_samples_harmful
        target_nonref = CFG.max_samples_benign
    print(f"[INFO] Target per class: {target_refusal} (unsafe) / {target_nonref} (safe)")

    data_iter = iter(ds_stream[split])

    refusal_pat = refusal_regex()
    n_refusal = 0
    n_nonref = 0
    data_type_counts: Dict[str, int] = {}

    sums_refusal: Dict[int, torch.Tensor] = {}
    sums_nonref: Dict[int, torch.Tensor] = {}
    embeddings_refusal: Dict[int, List[np.ndarray]] = {L: [] for L in layer_candidates}
    embeddings_nonref: Dict[int, List[np.ndarray]] = {L: [] for L in layer_candidates}

    processed = 0
    log_every = 25

    while (n_refusal < target_refusal) or (n_nonref < target_nonref):
        try:
            row = next(data_iter)
        except StopIteration:
            print("[WARN] Dataset iterator exhausted before reaching targets.")
            break

        dt = str(row.get("source_dataset", row.get("data_type", "unknown")))
        data_type_counts[dt] = data_type_counts.get(dt, 0) + 1

        user_text = pick_user_prompt(row)
        assistant_text = row.get("target", "") or row.get("completion", "")
        if not is_valid_completion(assistant_text, CFG.min_completion_chars):
            continue

        sl = row.get("safety_label")
        if sl in ("safe", "unsafe"):
            is_refusal = (sl == "unsafe")
        else:
            is_refusal = bool(refusal_pat.search(assistant_text))

        if is_refusal and n_refusal >= target_refusal:
            continue
        if (not is_refusal) and n_nonref >= target_nonref:
            continue

        try:
            per_layer = first_assistant_hidden_seq_for_layers(
                tok, model, user_text, assistant_text, layer_candidates, CFG.avg_k_first_assistant_tokens
            )
        except Exception as e:
            print(f"[WARN] Forward failed on example {processed}: {e}")
            continue

        if not per_layer:
            continue

        if not sums_refusal and not sums_nonref:
            for L, v in per_layer.items():
                zeros = torch.zeros_like(v, dtype=torch.float64, device="cpu")
                sums_refusal[L] = zeros.clone()
                sums_nonref[L] = zeros.clone()

        for L, v in per_layer.items():
            if L not in sums_refusal:
                sums_refusal[L] = torch.zeros_like(v, dtype=torch.float64, device="cpu")
                sums_nonref[L] = torch.zeros_like(v, dtype=torch.float64, device="cpu")
            v_cpu = v.to(dtype=torch.float64, device="cpu")
            if is_refusal:
                sums_refusal[L] += v_cpu
                if n_refusal < CFG.viz_samples_per_class:
                    embeddings_refusal[L].append(v.mean(dim=0).cpu().numpy())
            else:
                sums_nonref[L] += v_cpu
                if n_nonref < CFG.viz_samples_per_class:
                    embeddings_nonref[L].append(v.mean(dim=0).cpu().numpy())

        if is_refusal:
            n_refusal += 1
        else:
            n_nonref += 1
        processed += 1
        if processed % log_every == 0:
            print(f"[INFO] Processed={processed} | refusal={n_refusal}/{target_refusal}, nonref={n_nonref}/{target_nonref}")

        if (n_refusal >= target_refusal) and (n_nonref >= target_nonref):
            break

    if (n_refusal == 0) or (n_nonref == 0):
        print(f"[ERROR] Not enough examples gathered. Refusal={n_refusal}, Non-refusal={n_nonref}.")
        return

    # Compute means and per-layer sequences
    mean_refusal: Dict[int, torch.Tensor] = {}
    mean_nonref: Dict[int, torch.Tensor] = {}
    per_layer_seq: Dict[int, torch.Tensor] = {}
    for L in layer_candidates:
        r = (sums_refusal[L] / float(n_refusal)).to(torch.float32)
        g = (sums_nonref[L] / float(n_nonref)).to(torch.float32)
        mean_refusal[L] = r
        mean_nonref[L] = g
        per_layer_seq[L] = (r - g)

    # Collapsed vectors (mean over K)
    per_layer_vec: Dict[int, torch.Tensor] = {L: v.mean(dim=0) for L, v in per_layer_seq.items()}

    # Compute separation metrics and select best layer
    all_layer_embeddings = {
        L: {
            'refusal': embeddings_refusal[L][:CFG.viz_samples_per_class],
            'nonref': embeddings_nonref[L][:CFG.viz_samples_per_class],
        }
        for L in layer_candidates
        if len(embeddings_refusal[L]) > 1 and len(embeddings_nonref[L]) > 1
    }
    separation_metrics = compute_separation_metrics(all_layer_embeddings)
    if separation_metrics:
        best_layer_for_sep = max(separation_metrics.items(), key=lambda x: x[1]['separation_ratio'])[0]
    else:
        # Fallback to the last layer in the search range
        best_layer_for_sep = layer_candidates[-1]

    if CFG.enable_visualization and separation_metrics:
        print("\n[INFO] Creating latent space visualizations and cross-layer metrics...")
        viz_dir = os.path.join(CFG.out_dir, "visualizations_lastlayer")
        os.makedirs(viz_dir, exist_ok=True)
        # Per-layer visuals (optional)
        for L in sorted(all_layer_embeddings.keys()):
            if len(embeddings_refusal[L]) > 10 and len(embeddings_nonref[L]) > 10:
                visualizer = LatentSpaceVisualizer(
                    embeddings_refusal[L][:CFG.viz_samples_per_class],
                    embeddings_nonref[L][:CFG.viz_samples_per_class],
                    L,
                    viz_dir,
                )
                visualizer.create_all_visualizations()
        # Cross-layer metric plots
        plot_cross_layer_metrics(separation_metrics, viz_dir)
        print(f"[INFO] Visualizations saved to {viz_dir}")

        best_layers = sorted(separation_metrics.items(), key=lambda x: x[1]['separation_ratio'], reverse=True)[:5]
        print("\n[INFO] Top 5 layers by cluster separation ratio:")
        for layer, metrics in best_layers:
            print(f"  Layer {layer}: ratio={metrics['separation_ratio']:.3f}, inter={metrics['inter_dist']:.3f}, intra={metrics['avg_intra']:.3f}")

    # Persist outputs
    meta = {
        "config": asdict(CFG),
        "model_name": CFG.model_name,
        "num_layers": int(num_layers),
        "layer_candidates": layer_candidates,
        "counts": {"refusal": int(n_refusal), "non_refusal": int(n_nonref)},
        "data_type_counts": data_type_counts,
        "regex": None,
        "labeling": "ACCD safety_label (unsafe vs safe) when available, else regex",
        "vector_semantics": "unsafe_minus_safe at first assistant tokens",
        "selection": "best_layer_by_separation_ratio",
        "best_layer": int(best_layer_for_sep),
    }

    with open(os.path.join(CFG.out_dir, f"{CFG.out_prefix}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    best_layer_vector = per_layer_vec[int(best_layer_for_sep)]
    steering_vector = best_layer_vector

    # Prepare activations to store (capped by viz_samples_per_class)
    per_layer_activations_refusal: Dict[int, torch.Tensor] = {}
    per_layer_activations_nonref: Dict[int, torch.Tensor] = {}
    for L in layer_candidates:
        if len(embeddings_refusal[L]) > 0:
            per_layer_activations_refusal[L] = torch.tensor(np.stack(embeddings_refusal[L]), dtype=torch.float32)
        if len(embeddings_nonref[L]) > 0:
            per_layer_activations_nonref[L] = torch.tensor(np.stack(embeddings_nonref[L]), dtype=torch.float32)

    payload = {
        # Per-layer token sequences (K x d)
        "per_layer_seq": {int(L): v.cpu() for L, v in per_layer_seq.items()},
        # Collapsed per-layer vectors (d,)
        "per_layer": {int(L): v.cpu() for L, v in per_layer_vec.items()},
        # Best layer and vector
        "best_layer": int(best_layer_for_sep),
        "best_layer_vector": steering_vector.cpu(),
        # Steering vector alias points to the best layer
        "steering_vector": steering_vector.cpu(),
        # Group means for reference (K x d)
        "mean_refusal_seq": {int(L): v.cpu() for L, v in mean_refusal.items()},
        "mean_nonref_seq": {int(L): v.cpu() for L, v in mean_nonref.items()},
        # Per-layer activations (per-example, capped by viz_samples_per_class)
        "per_layer_activations": {
            "refusal": {int(L): T.cpu() for L, T in per_layer_activations_refusal.items()},
            "nonref": {int(L): T.cpu() for L, T in per_layer_activations_nonref.items()},
        },
        # Separation metrics
        "separation_metrics": separation_metrics,
    }

    torch.save(payload, os.path.join(CFG.out_dir, f"{CFG.out_prefix}.vectors.pt"))

    # Also write a small JSON with the chosen layer and vector summary
    with open(os.path.join(CFG.out_dir, f"{CFG.out_prefix}.steering_vector.json"), "w") as f:
        json.dump({
            "best_layer": int(best_layer_for_sep),
            "vector_norm": float(torch.norm(steering_vector).item()),
        }, f, indent=2)

    print("\n[DONE] Saved vectors to:")
    print(" -", os.path.join(CFG.out_dir, f"{CFG.out_prefix}.meta.json"))
    print(" -", os.path.join(CFG.out_dir, f"{CFG.out_prefix}.vectors.pt"))
    print(" -", os.path.join(CFG.out_dir, f"{CFG.out_prefix}.steering_vector.json"))
    if CFG.enable_visualization and separation_metrics:
        print(" -", os.path.join(CFG.out_dir, "visualizations_lastlayer/"))


if __name__ == "__main__":
    main()

