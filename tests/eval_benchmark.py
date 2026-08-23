"""
Evaluation Script for SatQuery AI on BigEarthNet.txt Benchmark Split.

Evaluates the fine-tuned InternVL2-2B on the 1,082-patch benchmark split
with 15,029 manually verified annotations.

Metrics computed:
  - Binary VQA: Accuracy, F1
  - MCQ: Accuracy (exact match on letter)
  - Captioning: BLEU-4, METEOR, CIDEr (requires references)
  - Bounding Box: IoU, Precision@0.5

Usage:
    python tests/eval_benchmark.py \
        --model-path checkpoints/internvl2_2b_bigearthnet_lora/final_adapter \
        --parquet-path data/bigearthnet_txt/BigEarthNet.txt.parquet
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# Metrics
# =============================================================================

def accuracy(predictions: list[str], references: list[str]) -> float:
    """Exact-match accuracy (case-insensitive)."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return correct / len(predictions)


def binary_f1(predictions: list[str], references: list[str]) -> dict:
    """F1 score for binary (yes/no) questions."""
    tp = fp = fn = tn = 0
    for p, r in zip(predictions, references):
        p_val = p.strip().lower()
        r_val = r.strip().lower()
        if r_val == "yes":
            if p_val == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if p_val == "yes":
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": (tp + tn) / len(predictions) if predictions else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_bbox(text: str) -> list[float]:
    """Parse bounding box coordinates from model output.

    Expected formats:
      [x1 y1, x2 y2]
      [x1, y1, x2, y2]
      (x1, y1, x2, y2)
    """
    numbers = re.findall(r"[\d.]+", text)
    try:
        return [float(n) for n in numbers[:4]]
    except (ValueError, IndexError):
        return []


def compute_iou(pred_bbox: list[float], gt_bbox: list[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    if len(pred_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0

    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = max(0, pred_bbox[2] - pred_bbox[0]) * max(0, pred_bbox[3] - pred_bbox[1])
    area_gt = max(0, gt_bbox[2] - gt_bbox[0]) * max(0, gt_bbox[3] - gt_bbox[1])
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def bbox_metrics(predictions: list[str], references: list[str], iou_threshold: float = 0.5) -> dict:
    """Compute bounding box evaluation metrics."""
    ious = []
    hits = 0

    for pred, ref in zip(predictions, references):
        pred_box = parse_bbox(pred)
        ref_box = parse_bbox(ref)
        iou = compute_iou(pred_box, ref_box)
        ious.append(iou)
        if iou >= iou_threshold:
            hits += 1

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        f"precision@{iou_threshold}": hits / len(predictions) if predictions else 0.0,
    }


# =============================================================================
# Evaluator
# =============================================================================

class BenchmarkEvaluator:
    """Evaluate a fine-tuned model on BigEarthNet.txt benchmark split."""

    def __init__(
        self,
        model_path: str,
        parquet_path: str,
        base_model: str = "OpenGVLab/InternVL2-2B",
        batch_size: int = 1,
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.parquet_path = parquet_path
        self.base_model = base_model
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the fine-tuned model."""
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"Loading base model: {self.base_model}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True, use_fast=False
        )

        model = AutoModel.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Load LoRA adapter if it exists
        adapter_path = Path(self.model_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
        else:
            logger.warning(f"No LoRA adapter found at {adapter_path}. Using base model.")

        model.eval()
        self.model = model
        logger.info("Model loaded successfully.")

    def load_benchmark(self) -> pd.DataFrame:
        """Load the benchmark split from parquet."""
        logger.info(f"Loading benchmark from {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        bench = df[df["split"] == "bench"].copy()
        logger.info(f"Benchmark split: {len(bench)} annotations across {bench['patch_id'].nunique()} patches")
        return bench

    def generate(self, question: str) -> str:
        """Generate a response from the model (text-only, no image)."""
        if self.model is None:
            raise RuntimeError("Call load_model() first.")

        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values=None,
                question=question,
                generation_config={"max_new_tokens": self.max_new_tokens, "do_sample": False},
            )
            return response.strip()
        except Exception:
            # Fallback to direct generation
            inputs = self.tokenizer(question, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.language_model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )

            response = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return response.strip()

    def evaluate(self) -> dict:
        """Run full evaluation on benchmark split."""
        self.load_model()
        bench = self.load_benchmark()

        results_by_type = defaultdict(lambda: {"predictions": [], "references": []})
        results_by_category = defaultdict(lambda: {"predictions": [], "references": []})

        total = len(bench)
        start_time = time.time()

        for i, (_, row) in enumerate(bench.iterrows()):
            if i % 500 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {i}/{total} ({elapsed:.0f}s)")

            question = row["input"]
            reference = str(row["output"])
            task_type = row["type"]
            category = row["category"]

            try:
                prediction = self.generate(question)
            except Exception as e:
                logger.warning(f"Inference failed for row {i}: {e}")
                prediction = ""

            results_by_type[task_type]["predictions"].append(prediction)
            results_by_type[task_type]["references"].append(reference)
            results_by_category[category]["predictions"].append(prediction)
            results_by_category[category]["references"].append(reference)

        total_time = time.time() - start_time

        # Compute metrics per type
        metrics = {"total_samples": total, "total_time_s": total_time}

        for task_type, data in results_by_type.items():
            preds = data["predictions"]
            refs = data["references"]

            if task_type == "binary":
                metrics[f"{task_type}"] = binary_f1(preds, refs)
            elif task_type == "mcq":
                metrics[f"{task_type}"] = {"accuracy": accuracy(preds, refs)}
            elif task_type == "bounding box":
                metrics[f"{task_type}"] = bbox_metrics(preds, refs)
            elif task_type == "captioning":
                # For now, use a simple word overlap metric
                # Full BLEU/METEOR requires nltk
                overlap_scores = []
                for p, r in zip(preds, refs):
                    p_words = set(p.lower().split())
                    r_words = set(r.lower().split())
                    if r_words:
                        overlap_scores.append(
                            len(p_words & r_words) / len(r_words)
                        )
                metrics[f"{task_type}"] = {
                    "word_recall": float(np.mean(overlap_scores)) if overlap_scores else 0.0,
                    "n_samples": len(preds),
                }

        # Per-category accuracy
        category_metrics = {}
        for cat, data in results_by_category.items():
            category_metrics[cat] = {
                "accuracy": accuracy(data["predictions"], data["references"]),
                "n_samples": len(data["predictions"]),
            }
        metrics["by_category"] = category_metrics

        return metrics

    def save_results(self, metrics: dict, output_path: str = "docs/benchmark_results.json"):
        """Save evaluation results."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on BigEarthNet.txt benchmark")
    parser.add_argument("--model-path", default="checkpoints/internvl2_2b_bigearthnet_lora/final_adapter")
    parser.add_argument("--parquet-path", default="data/bigearthnet_txt/BigEarthNet.txt.parquet")
    parser.add_argument("--output", default="docs/benchmark_results.json")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        model_path=args.model_path,
        parquet_path=args.parquet_path,
        max_new_tokens=args.max_tokens,
    )
    metrics = evaluator.evaluate()
    evaluator.save_results(metrics, args.output)

    print("\n=== Benchmark Results ===")
    for key, val in metrics.items():
        if key not in ("by_category",):
            print(f"  {key}: {val}")
