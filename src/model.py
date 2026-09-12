"""Tiny-H3 model construction: a small ``MiniMaxH3Transformer3DModel`` plus its text conditioner.

The transformer is diffusers' own MiniMax-H3 class with the dimensions turned down; the
latent shapes stay exactly H3's (24 video channels, 32 audio channels, ``(1, 2, 2)`` patch),
so the frozen official VAEs and this DiT speak the same latent language.

Presets (parameter counts are approximate, text encoder excluded)::

    smoke        ~9M   CPU smoke tests, 64x64 / 22 frames
    tiny_h3_5090 ~44M  the default single-5090 configuration, 256x256 / 22 frames
    tiny_h3_xl   ~110M 512x512 / 39 frames if you have the patience

Text conditioning replaces H3's 66 GB Qwen3-VL-8B with a frozen T5 (``t5-small``, 512-dim).
Prompts are padded to a fixed token count so every sample in a batch shares one packed
layout -- the transformer's batch axis is a pure replication axis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

from .vae import AUDIO_LATENT_CHANNELS, VIDEO_LATENT_CHANNELS

DEFAULT_TEXT_ENCODER = "Qwen/Qwen3-0.6B"
DEFAULT_TEXT_DIM = 1024  # Qwen3-0.6B hidden size
DEFAULT_TEXT_TOKENS = 48

# Small Qwen options, all frozen, all fine to run on one 5090:
#   Qwen/Qwen3-0.6B      1024-dim, 28 layers  (default; closest to official H3's Qwen3-VL lineage)
#   Qwen/Qwen2.5-0.5B     896-dim, 24 layers
#   t5-small              512-dim, 12 layers  (lightest; handy for CPU smoke tests)
TEXT_ENCODER_CHOICES = ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B", "t5-small", "t5-v1_1-small")

TINY_H3_PRESETS: dict[str, dict] = {
    "smoke": dict(
        hidden_size=256, num_layers=6, num_refiner_layers=1, ffn_dim=704,
        num_attention_heads=4, attention_head_dim=64, time_embed_hidden_dim=512, time_embed_dim=128,
        rope_freq_dim=8,
    ),
    "tiny_h3_5090": dict(
        hidden_size=512, num_layers=10, num_refiner_layers=2, ffn_dim=1408,
        num_attention_heads=8, attention_head_dim=64, time_embed_hidden_dim=1024, time_embed_dim=256,
        rope_freq_dim=8,
    ),
    "tiny_h3_xl": dict(
        hidden_size=768, num_layers=14, num_refiner_layers=2, ffn_dim=2048,
        num_attention_heads=12, attention_head_dim=64, time_embed_hidden_dim=1536, time_embed_dim=384,
        rope_freq_dim=8,
    ),
    # Still only ~3.2 GiB of parameters+optimizer states: the 5090 has room, the budget that
    # really runs out first is data and wall-clock.
    "tiny_h3_5090_max": dict(
        hidden_size=1024, num_layers=16, num_refiner_layers=2, ffn_dim=2816,
        num_attention_heads=16, attention_head_dim=64, time_embed_hidden_dim=2048, time_embed_dim=512,
        rope_freq_dim=8,
    ),
}
# Every preset uses head_dim 64; the official H3 pairs rope_freq_dim 16 with head_dim 128, i.e.
# rotary angles span 6*16 = 96 of 128 channels (75%).  rope_freq_dim=8 spans 48 of 64 -- the
# same fraction, and anything larger than 10 overruns the head_dim inside the H3 attention.


@dataclass
class ModelSpec:
    """What was trained, so a checkpoint can be rebuilt without guesswork."""

    preset: str = "tiny_h3_5090"
    text_encoder: str = DEFAULT_TEXT_ENCODER
    text_dim: int = DEFAULT_TEXT_DIM
    text_tokens: int = DEFAULT_TEXT_TOKENS
    text_layer: int = -1  # -1 = final hidden state; >=0 mimics H3's intermediate-layer read
    in_channels: int = VIDEO_LATENT_CHANNELS
    audio_in_channels: int = AUDIO_LATENT_CHANNELS
    patch_size: tuple[int, int, int] = (1, 2, 2)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "text_encoder": self.text_encoder,
            "text_dim": self.text_dim,
            "text_tokens": self.text_tokens,
            "text_layer": self.text_layer,
            "in_channels": self.in_channels,
            "audio_in_channels": self.audio_in_channels,
            "patch_size": list(self.patch_size),
            "preset_kwargs": TINY_H3_PRESETS[self.preset],
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        preset = d.get("preset", "tiny_h3_5090")
        # A checkpoint may carry dimensions that predate a preset tweak; register them so the
        # rebuild stays faithful even if the preset table changes later.
        if "preset_kwargs" in d:
            TINY_H3_PRESETS.setdefault(preset, dict(d["preset_kwargs"]))
        return cls(
            preset=preset,
            text_encoder=d.get("text_encoder", DEFAULT_TEXT_ENCODER),
            text_dim=int(d.get("text_dim", DEFAULT_TEXT_DIM)),
            text_tokens=int(d.get("text_tokens", DEFAULT_TEXT_TOKENS)),
            text_layer=int(d.get("text_layer", -1)),
            in_channels=int(d.get("in_channels", VIDEO_LATENT_CHANNELS)),
            audio_in_channels=int(d.get("audio_in_channels", AUDIO_LATENT_CHANNELS)),
            patch_size=tuple(d.get("patch_size", (1, 2, 2))),
        )


def build_dit(spec: ModelSpec, dtype: torch.dtype = torch.float32) -> "torch.nn.Module":
    """Instantiate the tiny H3 transformer."""
    from diffusers import MiniMaxH3Transformer3DModel

    if spec.preset not in TINY_H3_PRESETS:
        raise KeyError(f"unknown preset {spec.preset!r}; available: {sorted(TINY_H3_PRESETS)}")
    kwargs = dict(TINY_H3_PRESETS[spec.preset])
    model = MiniMaxH3Transformer3DModel(
        in_channels=spec.in_channels,
        audio_in_channels=spec.audio_in_channels,
        patch_size=tuple(spec.patch_size),
        text_dim=spec.text_dim,
        **kwargs,
    )
    return model.to(dtype=dtype)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------------------------------
# text conditioning
# --------------------------------------------------------------------------------------
class TextConditioner:
    """Frozen text encoder producing fixed-length, zero-padded prompt embeddings.

    MiniMax-H3 itself conditions on layer 50 of a Qwen3-VL-8B (66 GB).  Tiny-H3 follows the
    same lineage at 0.x B scale -- ``Qwen/Qwen3-0.6B`` by default -- and falls back to a
    decoder-only or encoder-only model transparently:

    * decoder-only (Qwen2/3): ``AutoModel`` returns the final hidden state, which is what the
      DiT's ``context_embedder`` consumes;
    * encoder-only (T5): ``T5EncoderModel`` last hidden state.

    Prompts are padded to ``max_tokens`` and the padding rows are zeroed, because the packed
    sequence has no attention mask and pad rows must be inert.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_ENCODER,
        max_tokens: int = DEFAULT_TEXT_TOKENS,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        text_layer: int = -1,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.text_layer = int(text_layer)
        self.device = torch.device(device)
        # Qwen tokenizers pad on the right by default; make it explicit for every family.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_tokens, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs = {"dtype": dtype}
        try:
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        except TypeError:
            # transformers < 5 spells the same argument `torch_dtype`.
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.text_dim = int(getattr(self.model.config, "hidden_size", None) or self.model.config.d_model)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def encode(self, prompts: list[str] | str) -> torch.Tensor:
        """``[prompt, ...]`` -> ``(B, max_tokens, text_dim)`` float32 embeddings.

        ``text_layer=-1`` uses the final hidden state.  Set an explicit index to mimic the
        official recipe, which conditions on an *intermediate* layer (MiniMax-H3 reads
        ``hidden_states[50]`` of its Qwen3-VL rather than the post-norm last one).
        """
        single = isinstance(prompts, str)
        prompts = [prompts] if single else list(prompts)
        batch = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        need_hidden = self.text_layer >= 0
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
        )
        hidden = out.hidden_states[self.text_layer] if need_hidden else out.last_hidden_state
        keep = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * keep).to(torch.float32).cpu()


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(
    out_dir: str,
    dit: torch.nn.Module,
    spec: ModelSpec,
    *,
    step: int | None = None,
    extra: dict | None = None,
) -> str:
    """Save the DiT (diffusers format) plus a ``tiny_h3_meta.json`` describing how to rebuild it."""
    os.makedirs(out_dir, exist_ok=True)
    dit.save_pretrained(out_dir, safe_serialization=True)
    meta = spec.to_dict()
    meta["step"] = step
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, "tiny_h3_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_dir


def load_dit(checkpoint_dir: str, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
    """Load a Tiny-H3 DiT and the spec it was trained with."""
    from diffusers import MiniMaxH3Transformer3DModel

    with open(os.path.join(checkpoint_dir, "tiny_h3_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    spec = ModelSpec.from_dict(meta)
    model = MiniMaxH3Transformer3DModel.from_pretrained(checkpoint_dir, torch_dtype=dtype)
    return model.to(device).eval(), spec


def load_text_conditioner_for(checkpoint_dir: str, device: str | torch.device = "cpu", dtype=torch.float32) -> TextConditioner:
    with open(os.path.join(checkpoint_dir, "tiny_h3_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return TextConditioner(
        meta.get("text_encoder", DEFAULT_TEXT_ENCODER),
        int(meta.get("text_tokens", DEFAULT_TEXT_TOKENS)),
        device,
        dtype,
        text_layer=int(meta.get("text_layer", -1)),
    )
