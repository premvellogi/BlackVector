"""
SatQuery AI — LoRA Training Script for InternVL2-2B.

This is the main training loop, validated by Phase 0 sanity check.
Trains the SpectralChannelAdapter + LoRA on the BigEarthNet.txt dataset.

Architecture (verified):
  S2 (10 bands) + S1 (2 bands)
    -> DualModalityAdapter (562 params, learned fusion)
    -> InternViT-300M (FROZEN, 4-bit)
    -> MLP Projector
    -> InternLM2-1.8B + LoRA (1,966,080 trainable params)
    -> Loss on assistant responses only

VRAM budget (RTX 3060, 6GB):
  Model (4-bit):     2.12 GB
  LoRA + adapter:    0.77 GB
  Forward pass:      ~2.5 GB
  Peak:              ~5.5 GB (gradient checkpointing ON)

Usage:
    # Phase 0: Quick sanity (already verified)
    python services/models/training/train_lora.py --max-steps 50 --eval-steps 25

    # Phase 1: Development run
    python services/models/training/train_lora.py --epochs 3 --eval-steps 500

    # Phase 2: Full training
    python services/models/training/train_lora.py --epochs 5 --eval-steps 1000
"""

import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery LoRA Training")
    parser.add_argument("--model-path", default="models/InternVL2-2B")
    parser.add_argument("--train-jsonl", default="data/training/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/training/val.jsonl")
    parser.add_argument("--imagery-cache", default="data/imagery_cache")
    parser.add_argument("--output-dir", default="checkpoints/satquery-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="0=use epochs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8, help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--s2-channels", type=int, default=10)
    parser.add_argument("--s1-channels", type=int, default=2)
    parser.add_argument(
        "--allow-synthetic-images",
        action="store_true",
        help=(
            "Use generated placeholder imagery for a short plumbing smoke test. "
            "Never use this for a checkpoint presented as RS-adapted."
        ),
    )
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save training config
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # =========================================================================
    # 1. Load model + tokenizer
    # =========================================================================
    logger.info("Loading InternVL2-2B in 4-bit quantization...")

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False,
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info(f"Model loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    # =========================================================================
    # 2. Attach LoRA
    # =========================================================================
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Freeze vision encoder
    for param in model.vision_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["wqkv", "wo"],  # InternLM2 attention modules
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA attached: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    # =========================================================================
    # 3. Create spectral adapter
    # =========================================================================
    from services.models.core.spectral_adapter import DualModalityAdapter

    adapter = DualModalityAdapter(
        s2_channels=args.s2_channels,
        s1_channels=args.s1_channels,
        target_size=448,
    ).to("cuda", dtype=torch.bfloat16)

    logger.info(f"Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # =========================================================================
    # 4. Load dataset
    # =========================================================================
    from services.models.training.multiband_dataset import BigEarthNetMultiBandDataset

    train_ds = BigEarthNetMultiBandDataset(
        jsonl_path=args.train_jsonl,
        imagery_cache_dir=args.imagery_cache,
        tokenizer=None,  # We tokenize manually in the training loop
        require_real_images=not args.allow_synthetic_images,
    )

    val_ds = BigEarthNetMultiBandDataset(
        jsonl_path=args.val_jsonl,
        imagery_cache_dir=args.imagery_cache,
        tokenizer=None,
        require_real_images=not args.allow_synthetic_images,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    logger.info(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")
    logger.info(f"Real imagery: {train_ds.get_real_image_stats()}")

    # =========================================================================
    # 5. Setup optimizer and scheduler
    # =========================================================================
    all_trainable = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(adapter.parameters())
    )

    optimizer = torch.optim.AdamW(all_trainable, lr=args.lr, weight_decay=0.01)

    total_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # =========================================================================
    # 6. Token setup
    # =========================================================================
    IMG_CTX = "<IMG_CONTEXT>"
    IMG_START = "<img>"
    IMG_END = "</img>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)
    vit_dtype = next(model.vision_model.parameters()).dtype
    num_image_tokens = 256  # InternVL2-2B default

    image_tokens_str = IMG_START + IMG_CTX * num_image_tokens + IMG_END

    # =========================================================================
    # 7. Training loop
    # =========================================================================
    logger.info(f"Starting training: {total_steps} steps, lr={args.lr}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")

    model.train()
    adapter.train()

    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    start_time = time.time()

    for epoch in range(args.epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # --- Build text with image tokens ---
            human_msg = batch["human_message"][0]
            gpt_msg = batch["assistant_message"][0]

            # Replace <image> placeholder with actual image tokens
            if "<image>" in human_msg:
                human_msg = human_msg.replace("<image>", image_tokens_str)
            else:
                human_msg = image_tokens_str + "\n" + human_msg

            query = (
                f"<|im_start|>user\n{human_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{gpt_msg}<|im_end|>"
            )

            inputs = tokenizer(
                query, return_tensors="pt",
                max_length=args.max_length, truncation=True,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            # Build labels (mask everything before assistant response)
            labels = input_ids.clone()
            assistant_header = tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            seq = input_ids[0].tolist()
            for pos in range(len(seq) - len(assistant_header)):
                if seq[pos:pos + len(assistant_header)] == assistant_header:
                    labels[0, :pos + len(assistant_header)] = -100
                    break

            # --- Forward: adapter -> ViT -> inject -> LLM ---
            s2_bands = batch["s2_bands"].to("cuda", dtype=torch.bfloat16)
            s1_bands = batch["s1_bands"].to("cuda", dtype=torch.bfloat16)

            # Spectral adapter
            pseudo_rgb = adapter(s2_bands, s1_bands)
            pseudo_rgb_cast = pseudo_rgb.to(dtype=vit_dtype)

            # ViT parameters are frozen, but this pass must retain a graph so
            # the loss can update the spectral-adapter input.
            vit_embeds = model.extract_feature(pseudo_rgb_cast)

            # Inject ViT features into embeddings
            input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
            B, N, C = input_embeds.shape
            input_embeds_flat = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == img_context_token_id)

            if selected.sum() > 0:
                vit_flat = vit_embeds.reshape(-1, C).to(input_embeds_flat.dtype)
                n_ctx = selected.sum().item()
                n_vit = vit_flat.shape[0]
                if n_ctx == n_vit:
                    input_embeds_flat[selected] = vit_flat
                else:
                    # Truncation: use min of available
                    n_use = min(n_ctx, n_vit)
                    indices = selected.nonzero(as_tuple=True)[0][:n_use]
                    input_embeds_flat[indices] = vit_flat[:n_use]

            input_embeds = input_embeds_flat.reshape(B, N, C)

            # Language model forward
            outputs = model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / args.grad_accum
            loss.backward()

            train_losses.append(loss.item() * args.grad_accum)
            epoch_loss += loss.item() * args.grad_accum
            epoch_steps += 1

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_interval == 0:
                    avg_loss = sum(train_losses[-args.log_interval:]) / args.log_interval
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    eta = elapsed / global_step * (total_steps - global_step) if global_step > 0 else 0
                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
                        f"ETA: {eta/60:.0f}min"
                    )

                # Evaluate
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(
                        model, adapter, val_loader, val_ds, tokenizer,
                        img_context_token_id, vit_dtype, image_tokens_str,
                        args.max_length,
                    )
                    logger.info(f"  Val loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, adapter, optimizer, global_step, args.output_dir, "best")
                        logger.info(f"  New best! Saved checkpoint.")

                    model.train()
                    adapter.train()

                # Save periodic checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, adapter, optimizer, global_step, args.output_dir, f"step-{global_step}")

                # Early stop
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        # Do not discard gradients from a final partial accumulation window.
        if epoch_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        logger.info(f"Epoch {epoch + 1} complete | Avg loss: {avg_epoch_loss:.4f}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save
    save_checkpoint(model, adapter, optimizer, global_step, args.output_dir, "final")

    # Final evaluation
    val_loss = evaluate(
        model, adapter, val_loader, val_ds, tokenizer,
        img_context_token_id, vit_dtype, image_tokens_str,
        args.max_length,
    )
    logger.info(f"\nTraining complete!")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Final val loss: {val_loss:.4f}")
    logger.info(f"  Adapter fusion: {adapter.modality_weight}")
    logger.info(f"  Saved to: {args.output_dir}")


def evaluate(model, adapter, val_loader, val_ds, tokenizer,
             img_context_token_id, vit_dtype, image_tokens_str, max_length,
             max_batches=50):
    """Run evaluation on validation set."""
    model.eval()
    adapter.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            human_msg = batch["human_message"][0]
            gpt_msg = batch["assistant_message"][0]

            if "<image>" in human_msg:
                human_msg = human_msg.replace("<image>", image_tokens_str)
            else:
                human_msg = image_tokens_str + "\n" + human_msg

            query = (
                f"<|im_start|>user\n{human_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{gpt_msg}<|im_end|>"
            )

            inputs = tokenizer(query, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            labels = input_ids.clone()
            assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            seq = input_ids[0].tolist()
            for pos in range(len(seq) - len(assistant_header)):
                if seq[pos:pos + len(assistant_header)] == assistant_header:
                    labels[0, :pos + len(assistant_header)] = -100
                    break

            s2 = batch["s2_bands"].to("cuda", dtype=torch.bfloat16)
            s1 = batch["s1_bands"].to("cuda", dtype=torch.bfloat16)

            pseudo_rgb = adapter(s2, s1).to(dtype=vit_dtype)
            vit_embeds = model.extract_feature(pseudo_rgb)

            input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
            B, N, C = input_embeds.shape
            flat = input_embeds.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            sel = (ids_flat == img_context_token_id)
            if sel.sum() > 0:
                vit_flat = vit_embeds.reshape(-1, C).to(flat.dtype)
                n_use = min(sel.sum().item(), vit_flat.shape[0])
                indices = sel.nonzero(as_tuple=True)[0][:n_use]
                flat[indices] = vit_flat[:n_use]
            input_embeds = flat.reshape(B, N, C)

            outputs = model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def save_checkpoint(model, adapter, optimizer, step, output_dir, name):
    """Save LoRA weights + adapter weights."""
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(ckpt_dir)

    # Save spectral adapter
    torch.save(adapter.state_dict(), os.path.join(ckpt_dir, "spectral_adapter.pt"))

    # Save optimizer state
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "fusion_weights": adapter.modality_weight,
    }, os.path.join(ckpt_dir, "training_state.pt"))

    logger.info(f"Checkpoint saved: {ckpt_dir}")


if __name__ == "__main__":
    main()
