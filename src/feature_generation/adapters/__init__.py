"""Pretrained model adapters; heavyweight backends are loaded only on demand."""

from .base import EmbeddingAdapter
from .esm2 import ESM2Adapter

__all__ = ["EmbeddingAdapter", "ESM2Adapter"]
