"""
SatQuery AI -- Exp 1 vs Exp 2 Comprehensive Benchmark.

Tests BOTH checkpoints on the SAME 20 unseen patches across:
  1. VQA accuracy (binary yes/no questions with known ground truth)
  2. Caption quality (length, detail, land-cover keyword coverage)
  3. SAR vs Optical modality attribution (fusion analysis)
  4. Change detection (same-patch vs different-patch discrimination)

Each experiment is loaded, tested, unloaded -- then results are compared.

Usage:
    python tests/benchmark_exp1_vs_exp2.py
"""

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────

EXPERIMENTS = {
    "exp1": {
        "adapter": "checkpoints/satquery-lora-exp1/best",
        "label": "Exp 1 (494 patches)",
    },
    "exp2": {
        "adapter": "checkpoints/satquery-lora-exp2/best",
        "label": "Exp 2 (1500 patches)",
    },
}

N_BENCHMARK_PATCHES = 20
N_VQA_PER_PATCH = 3        # binary questions per patch
N_CHANGE_PAIRS = 5         # change detection pairs
PARQUET_PATH = "data/bigearthnet_txt/BigEarthNet.txt.parquet"
IMAGERY_CACHE = "data/imagery_cache"
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────

def get_unseen_patches():
    """Find patches unseen by both experiments."""
    exp1_patches = set()
    for f in ["data/training_exp1/train.jsonl", "data/training_exp1/val.jsonl"]:
        if os.path.exists(f):
            for line in open(f):
                exp1_patches.add(json.loads(line)["image"])

    exp2_patches = set()
    for f in ["data/training_exp2/train.jsonl", "data/training_exp2/val.jsonl"]:
        if os.path.exists(f):
            for line in open(f):
                exp2_patches.add(json.loads(line)["image"])

    cached = set(p.stem for p in Path(IMAGERY_CACHE).glob("*.npz"))
    unseen = sorted(cached - exp1_patches - exp2_patches)

    random.seed(SEED)
    sample = random.sample(unseen, min(N_BENCHMARK_PATCHES, len(unseen)))
    return sorted(sample)


def load_patch(patch_id):
    """Load a patch's S2 and S1 bands."""
    from services.models.training.multiband_dataset import S2_BANDS, S2_STATS, S1_STATS

    s2_mean = np.array([S2_STATS[b]["mean"] for b in S2_BANDS]).reshape(-1, 1, 1)
    s2_std = np.array([S2_STATS[b]["std"] for b in S2_BANDS]).reshape(-1, 1, 1)
    s1_mean = np.array([S1_STATS["VV"]["mean"], S1_STATS["VH"]["mean"]]).reshape(-1, 1, 1)
    s1_std = np.array([S1_STATS["VV"]["std"], S1_STATS["VH"]["std"]]).reshape(-1, 1, 1)

    data = np.load(os.path.join(IMAGERY_CACHE, f"{patch_id}.npz"))
    s2 = data["s2"].astype(np.float32)
    s1 = data.get("s1", np.zeros((2, 120, 120), dtype=np.float32)).astype(np.float32)

    s2_norm = (s2 - s2_mean) / (s2_std + 1e-8)
    s1_norm = (s1 - s1_mean) / (s1_std + 1e-8)

    s2_t = torch.from_numpy(s2_norm).unsqueeze(0)
    s1_t = torch.from_numpy(s1_norm).unsqueeze(0)
    return s2_t, s1_t


def get_ground_truth_qa(df, patch_ids):
    """Get binary QA ground truth for benchmark patches."""
    qa_pairs = []
    for pid in patch_ids:
        subset = df[(df["patch_id"] == pid) & (df["type"] == "binary")]
        if len(subset) == 0:
            continue
        sampled = subset.sample(n=min(N_VQA_PER_PATCH, len(subset)), random_state=SEED)
        for _, row in sampled.iterrows():
            qa_pairs.append({
                "patch_id": pid,
                "question": row["input"],
                "answer": row["output"].strip().lower(),
            })
    return qa_pairs


# ── Benchmark Runner ─────────────────────────────────────────────────────

def run_benchmark_for_experiment(exp_name, exp_config, patches, qa_pairs, df):
    """Run full benchmark suite for one experiment."""
    from services.models.core.vlm_backbone import VLMBackbone, ModelConfig
    from services.models.heads.change_head import ChangeDetector
    from services.models.heads.fusion_head import FusionAnalyzer

    print(f"\n{'='*65}")
    print(f"  BENCHMARKING: {exp_config['label']}")
    print(f"  Adapter: {exp_config['adapter']}")
    print(f"{'='*65}\n")

    # Load model
    t0 = time.time()
    config = ModelConfig()
    config.lora_adapter_path = exp_config["adapter"]
    vlm = VLMBackbone(config)
    vlm.load()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s\n")

    results = {
        "experiment": exp_name,
        "label": exp_config["label"],
        "adapter": exp_config["adapter"],
        "load_time": round(load_time, 1),
    }

    # ── 1. VQA Accuracy ──
    print("  [1/4] VQA Accuracy (binary yes/no)...")
    vqa_correct = 0
    vqa_total = 0
    vqa_details = []

    for qa in qa_pairs:
        s2_t, s1_t = load_patch(qa["patch_id"])
        s2_t = s2_t.to(vlm._device, dtype=torch.bfloat16)
        s1_t = s1_t.to(vlm._device, dtype=torch.bfloat16)

        prompt = (
            f"You are a remote sensing expert. "
            f"Answer with only 'yes' or 'no'.\n"
            f"Question: {qa['question']}"
        )
        result = vlm.generate_from_bands(s2_t, s1_t, prompt, max_new_tokens=5)
        answer = result["text"].strip().lower()
        # Normalize answer
        pred = "yes" if "yes" in answer else ("no" if "no" in answer else answer)
        gt = qa["answer"]
        correct = pred == gt

        vqa_correct += int(correct)
        vqa_total += 1
        vqa_details.append({
            "patch": qa["patch_id"][-10:],
            "pred": pred,
            "gt": gt,
            "correct": correct,
        })

    vqa_acc = vqa_correct / max(vqa_total, 1)
    results["vqa_accuracy"] = round(vqa_acc, 4)
    results["vqa_correct"] = vqa_correct
    results["vqa_total"] = vqa_total
    print(f"    Accuracy: {vqa_correct}/{vqa_total} ({vqa_acc*100:.1f}%)\n")

    # ── 2. Caption Quality ──
    print("  [2/4] Caption Quality...")
    caption_lengths = []
    caption_keywords = []

    LAND_COVER_KEYWORDS = [
        "urban", "forest", "agricultural", "water", "residential",
        "vegetation", "road", "building", "crop", "grass", "industrial",
        "river", "lake", "field", "meadow", "pasture", "woodland",
    ]

    for pid in patches[:10]:  # 10 patches for captioning
        s2_t, s1_t = load_patch(pid)
        s2_t = s2_t.to(vlm._device, dtype=torch.bfloat16)
        s1_t = s1_t.to(vlm._device, dtype=torch.bfloat16)

        prompt = (
            "You are a remote sensing expert. "
            "Describe this satellite image in detail, including land cover types, "
            "spatial arrangement, and notable features."
        )
        result = vlm.generate_from_bands(s2_t, s1_t, prompt, max_new_tokens=512)
        caption = result["text"]
        caption_lengths.append(len(caption))

        kw_count = sum(1 for kw in LAND_COVER_KEYWORDS if kw in caption.lower())
        caption_keywords.append(kw_count)

    results["caption_avg_length"] = round(np.mean(caption_lengths), 0)
    results["caption_min_length"] = min(caption_lengths)
    results["caption_max_length"] = max(caption_lengths)
    results["caption_avg_keywords"] = round(np.mean(caption_keywords), 2)
    print(f"    Avg length: {results['caption_avg_length']:.0f} chars")
    print(f"    Range: [{results['caption_min_length']}, {results['caption_max_length']}]")
    print(f"    Avg land-cover keywords: {results['caption_avg_keywords']:.1f}\n")

    # ── 3. SAR vs Optical Fusion ──
    print("  [3/4] SAR vs Optical Fusion Attribution...")
    fusion = FusionAnalyzer(vlm)
    optical_contributions = []
    sar_contributions = []
    modality_distances = []
    fusion_total = 0

    for pid in patches[:8]:  # 8 patches for fusion
        s2_t, s1_t = load_patch(pid)
        s2_t = s2_t.to(vlm._device, dtype=torch.bfloat16)
        s1_t = s1_t.to(vlm._device, dtype=torch.bfloat16)

        try:
            result = fusion.analyze(
                s2_t, s1_t,
                question="What land cover types are visible in this image?"
            )

            # New v2 API: embedding-based cosine attribution
            optical_contributions.append(result["optical_contribution"])
            sar_contributions.append(result["sar_contribution"])
            modality_distances.append(result["modality_distance"])
            fusion_total += 1
        except Exception as e:
            print(f"    WARN: Fusion failed for {pid[-10:]}: {e}")

    results["fusion_optical_contrib"] = round(np.mean(optical_contributions), 4) if optical_contributions else 0
    results["fusion_sar_contrib"] = round(np.mean(sar_contributions), 4) if sar_contributions else 0
    results["fusion_modality_distance"] = round(np.mean(modality_distances), 4) if modality_distances else 0
    results["fusion_total"] = fusion_total
    results["fusion_disagreements"] = sum(1 for d in modality_distances if d > 0.3)
    print(f"    Avg optical contribution: {results['fusion_optical_contrib']:.1%}")
    print(f"    Avg SAR contribution:     {results['fusion_sar_contrib']:.1%}")
    print(f"    Avg modality distance:    {results['fusion_modality_distance']:.4f}")
    print(f"    Disagreements detected:   {results['fusion_disagreements']}/{fusion_total}\n")

    # ── 4. Change Detection ──
    print("  [4/4] Change Detection...")
    detector = ChangeDetector(vlm)

    # Same-patch tests (should score ~0)
    same_scores = []
    for pid in patches[:N_CHANGE_PAIRS]:
        s2_t, s1_t = load_patch(pid)
        s2_t = s2_t.to(vlm._device, dtype=torch.bfloat16)
        s1_t = s1_t.to(vlm._device, dtype=torch.bfloat16)
        result = detector.detect_change(s2_t, s1_t, s2_t, s1_t)
        same_scores.append(result["change_score"])

    # Different-patch tests (should score ~1)
    diff_scores = []
    for i in range(N_CHANGE_PAIRS):
        p1, p2 = patches[i], patches[-(i+1)]
        s2_a, s1_a = load_patch(p1)
        s2_b, s1_b = load_patch(p2)
        s2_a = s2_a.to(vlm._device, dtype=torch.bfloat16)
        s1_a = s1_a.to(vlm._device, dtype=torch.bfloat16)
        s2_b = s2_b.to(vlm._device, dtype=torch.bfloat16)
        s1_b = s1_b.to(vlm._device, dtype=torch.bfloat16)
        result = detector.detect_change(s2_a, s1_a, s2_b, s1_b)
        diff_scores.append(result["change_score"])

    results["change_same_mean"] = round(np.mean(same_scores), 4)
    results["change_diff_mean"] = round(np.mean(diff_scores), 4)
    results["change_separation"] = round(np.mean(diff_scores) - np.mean(same_scores), 4)
    change_correct = sum(1 for s in same_scores if s < 0.5) + sum(1 for s in diff_scores if s > 0.5)
    change_total = len(same_scores) + len(diff_scores)
    results["change_accuracy"] = f"{change_correct}/{change_total}"
    print(f"    Same-patch mean:  {results['change_same_mean']:.4f}")
    print(f"    Diff-patch mean:  {results['change_diff_mean']:.4f}")
    print(f"    Separation:       {results['change_separation']:.4f}")
    print(f"    Accuracy:         {change_correct}/{change_total}\n")

    # Unload
    vlm.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SatQuery AI -- Experiment 1 vs Experiment 2 Benchmark")
    print("=" * 65)

    # Get unseen patches
    patches = get_unseen_patches()
    print(f"\n  Benchmark patches: {len(patches)} (unseen by both experiments)")

    # Load ground truth
    print("  Loading ground truth annotations...")
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["split"] == "train"]
    qa_pairs = get_ground_truth_qa(df, patches)
    print(f"  VQA test pairs: {len(qa_pairs)}\n")

    # Run benchmarks
    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        results = run_benchmark_for_experiment(exp_name, exp_config, patches, qa_pairs, df)
        all_results[exp_name] = results

    # ── Comparison Table ──
    exp1 = all_results["exp1"]
    exp2 = all_results["exp2"]

    print("\n" + "=" * 65)
    print("  EXPERIMENT 1 vs EXPERIMENT 2 COMPARISON")
    print("=" * 65)
    print(f"\n  {'Metric':<30} {'Exp 1':>15} {'Exp 2':>15} {'Winner':>10}")
    print(f"  {'-'*70}")

    def compare(label, v1, v2, higher_is_better=True):
        s1 = str(v1)
        s2 = str(v2)
        try:
            f1, f2 = float(v1), float(v2)
            if higher_is_better:
                winner = "Exp 1" if f1 > f2 else ("Exp 2" if f2 > f1 else "Tie")
            else:
                winner = "Exp 1" if f1 < f2 else ("Exp 2" if f2 < f1 else "Tie")
        except (ValueError, TypeError):
            winner = "--"
        print(f"  {label:<30} {s1:>15} {s2:>15} {winner:>10}")

    compare("VQA Accuracy", f"{exp1['vqa_accuracy']*100:.1f}%", f"{exp2['vqa_accuracy']*100:.1f}%")
    compare("Caption Avg Length", exp1["caption_avg_length"], exp2["caption_avg_length"])
    compare("Caption Avg Keywords", exp1["caption_avg_keywords"], exp2["caption_avg_keywords"])
    compare("Optical Contribution", exp1["fusion_optical_contrib"], exp2["fusion_optical_contrib"])
    compare("SAR Contribution", exp1["fusion_sar_contrib"], exp2["fusion_sar_contrib"])
    compare("Modality Distance", exp1["fusion_modality_distance"], exp2["fusion_modality_distance"])
    compare("Change Same Score", exp1["change_same_mean"], exp2["change_same_mean"], higher_is_better=False)
    compare("Change Diff Score", exp1["change_diff_mean"], exp2["change_diff_mean"])
    compare("Change Separation", exp1["change_separation"], exp2["change_separation"])
    compare("Change Accuracy", exp1["change_accuracy"], exp2["change_accuracy"])

    # Save report
    os.makedirs("checkpoints", exist_ok=True)
    report_path = "checkpoints/exp1_vs_exp2_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
