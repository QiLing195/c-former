from __future__ import annotations

from dataclasses import dataclass

import torch


OBSERVER_TEXTS = {
    0: "look left",
    1: "look center",
    2: "look right",
}


@dataclass(frozen=True)
class SyntheticViewTask:
    """Generate sequences where the observer selects which token is the label."""

    seq_len: int = 6
    token_vocab_size: int = 16
    observer_vocab_size: int = 5

    # 0 is padding; remaining ids encode: look, left, center, right.
    observer_token_table = (
        (1, 2),
        (1, 3),
        (1, 4),
    )

    def selected_positions(self, observer_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.tensor(
            [0, self.seq_len // 2, self.seq_len - 1],
            device=observer_ids.device,
        )
        return positions[observer_ids]

    def observer_tokens(self, observer_ids: torch.Tensor) -> torch.Tensor:
        table = torch.tensor(self.observer_token_table, device=observer_ids.device)
        return table[observer_ids]

    def labels(self, tokens: torch.Tensor, observer_ids: torch.Tensor) -> torch.Tensor:
        positions = self.selected_positions(observer_ids)
        return tokens.gather(1, positions[:, None]).squeeze(1)

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.randint(
            self.token_vocab_size,
            (batch_size, self.seq_len),
            device=device,
            generator=generator,
        )
        observer_ids = torch.randint(3, (batch_size,), device=device, generator=generator)
        observer_tokens = self.observer_tokens(observer_ids)
        labels = self.labels(tokens, observer_ids)
        return tokens, observer_tokens, observer_ids, labels

