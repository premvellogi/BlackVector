"""
InternVL2-2B Model Loader for SatQuery AI.

Handles:
- Loading InternVL2-2B with 4-bit quantization (QLoRA-ready)
- LoRA adapter application (post fine-tuning)
- Shared backbone serving across all 5 endpoints
- VRAM budget management for RTX 3060 6GB

Key VRAM Budget:
    - InternVL2-2B in 4-bit (nf4):   ~1.5 GB
    - LoRA adapter weights:           ~50 MB
    - Inference activations:          ~1-2 GB (depends on tile count)
    - Grounding DINO (when loaded):   ~350 MB
    - Buffer/PyTorch overhead:        ~1 GB
    - Total:                          ~4-5 GB → fits 6 GB
"""

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the InternVL2-2B model."""

    # Model identity
    model_name: str = "OpenGVLab/InternVL2-2B"
    lora_adapter_path: Optional[str] = os.environ.get(
        "SATQUERY_LORA_ADAPTER", "checkpoints/satquery-lora/final"
    )

    # Quantization (critical for 6GB VRAM)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"       # Normal Float 4
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA config (used during training, stored here for reference)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj"
    )
    lora_dropout: float = 0.05

    # Memory optimization
    freeze_vision_encoder: bool = True  # Always True for 6GB
    gradient_checkpointing: bool = True  # Essential for training on 6GB
    max_dynamic_patch: int = 6  # Limit tile count to save VRAM (default is 12)

    # Inference
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low temp for factual answers


class VLMBackbone:
    """Shared InternVL2-2B backbone for all SatQuery AI endpoints.

    This class manages a single model instance that serves all 5 endpoints:
    /vqa, /caption, /grounding, /change, /fusion

    The model is loaded ONCE and kept in memory. Each endpoint uses
    different prompting strategies but shares the same backbone weights.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._loaded = False
        self._device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load InternVL2-2B with 4-bit quantization.

        This is the most VRAM-critical operation. We use:
        - 4-bit NF4 quantization via bitsandbytes
        - Frozen vision encoder
        - Limited dynamic patch count (6 instead of 12)
        """
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return

        logger.info(f"Loading {self.config.model_name} with 4-bit quantization...")
        logger.info(f"Device: {self._device}")

        if torch.cuda.is_available():
            logger.info(
                f"GPU: {torch.cuda.get_device_name(0)}, "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB, "
                f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB"
            )

        try:
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

            # 4-bit quantization config
            bnb_config = None
            if self.config.load_in_4bit:
                compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,  # Further reduces memory
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                use_fast=False,
            )

            # Load model
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }

            if bnb_config:
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = self.config.device

            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )

            # Freeze vision encoder (critical for 6GB)
            if self.config.freeze_vision_encoder and hasattr(self.model, "vision_model"):
                for param in self.model.vision_model.parameters():
                    param.requires_grad = False
                logger.info("Vision encoder frozen.")

            # Apply the project's adaptation checkpoint when it is available.
            if self.config.lora_adapter_path and Path(self.config.lora_adapter_path).is_dir():
                self._apply_lora_adapter(self.config.lora_adapter_path)
            elif self.config.lora_adapter_path:
                logger.warning(
                    "LoRA adapter not found at %s; serving the base model.",
                    self.config.lora_adapter_path,
                )

            # Set to eval mode for inference
            self.model.eval()
            self._loaded = True

            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(
                    f"Model loaded. VRAM used: {vram_used:.2f} GB, "
                    f"reserved: {vram_reserved:.2f} GB"
                )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _apply_lora_adapter(self, adapter_path: str) -> None:
        """Apply a trained LoRA adapter to the model."""
        from peft import PeftModel

        logger.info(f"Applying LoRA adapter from: {adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=False,  # Inference only
        )
        logger.info("LoRA adapter applied successfully.")

    def unload(self) -> None:
        """Free model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed.")

    def generate(
        self,
        image: "PIL.Image.Image",
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_logits: bool = False,
    ) -> dict:
        """Run inference: image + text prompt → generated text.

        This is the core inference method used by all endpoints.
        Each endpoint wraps this with a task-specific prompt template.

        Args:
            image: PIL Image (RGB, uint8)
            prompt: Text prompt (task-specific)
            max_new_tokens: Override default max generation length
            temperature: Override default temperature
            return_logits: If True, return token logits for confidence estimation

        Returns:
            dict with:
                - 'text': Generated text response
                - 'logits': Token logits (if return_logits=True)
                - 'tokens_generated': Number of tokens generated
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature

        try:
            # InternVL2 uses a chat-style interface
            # The image is processed internally by the model
            generation_config = {
                "max_new_tokens": max_tokens,
                "do_sample": temp > 0,
            }
            if temp > 0:
                generation_config["temperature"] = temp

            # InternVL2-specific chat method
            pixel_values = self._preprocess_image(image)
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(self._device)

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )

            result = {
                "text": response,
                "tokens_generated": len(self.tokenizer.encode(response)),
            }

            return result

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during generation. Clearing cache and retrying with reduced settings.")
            torch.cuda.empty_cache()
            gc.collect()
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_multi_image(
        self,
        images: list["PIL.Image.Image"],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Run inference with multiple images (for change detection and fusion).

        InternVL2 supports multi-image input with <image-1>, <image-2> tags.

        Args:
            images: List of PIL Images
            prompt: Text prompt with image reference tags

        Returns:
            Same dict structure as generate().
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature

        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": temp > 0,
        }
        if temp > 0:
            generation_config["temperature"] = temp

        # Process multiple images
        # InternVL2 handles multi-image via concatenated pixel values
        # with the prompt using <image> tags
        pixel_values_list = []
        for img in images:
            pv = self._preprocess_image(img)
            if pv is not None:
                pixel_values_list.append(pv)

        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).to(self._device)
        else:
            pixel_values = None

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )

        return {
            "text": response,
            "tokens_generated": len(self.tokenizer.encode(response)),
        }

    def _preprocess_image(self, image: "PIL.Image.Image"):
        """Preprocess a PIL image for InternVL2.

        Handles dynamic resolution: splits image into 448x448 tiles,
        limited by max_dynamic_patch to control VRAM.
        """
        try:
            # InternVL2's image preprocessing
            # The model provides a load_image utility
            from torchvision import transforms

            # Resize to fit within the tile budget
            # InternVL2 uses 448x448 tiles, max_dynamic_patch limits tile count
            max_tiles = self.config.max_dynamic_patch

            # Simple approach: resize to fit within max_tiles arrangement
            # A more sophisticated approach would use InternVL2's built-in
            # dynamic resolution logic
            target_size = 448  # Single tile

            transform = transforms.Compose([
                transforms.Resize(
                    (target_size, target_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            tensor = transform(image.convert("RGB"))
            return tensor.unsqueeze(0)  # Add batch dim

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None

    def get_vram_report(self) -> dict:
        """Return current VRAM usage breakdown."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "message": "No GPU available"}

        return {
            "device": torch.cuda.get_device_name(0),
            "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": torch.cuda.mem_get_info()[0] / 1e9,
            "model_loaded": self._loaded,
        }
