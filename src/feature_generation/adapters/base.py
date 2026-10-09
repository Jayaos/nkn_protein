"""Small interface shared by frozen residue-embedding extractors."""

from collections.abc import Iterator, Sequence
from typing import Protocol

from torch import Tensor


class EmbeddingAdapter(Protocol):
    @property
    def hidden_size(self) -> int: ...

    def metadata(self) -> dict: ...

    def load(self) -> None: ...

    def iter_embeddings(self, sequences: Sequence[str]) -> Iterator[tuple[list[int], Tensor]]: ...
