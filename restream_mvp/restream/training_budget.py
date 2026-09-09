"""Keep batch traversal and effective AdamW updates separate, including after resume."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingBudget:
    max_steps: int | None = None
    max_updates: int | None = None

    def __post_init__(self):
        if (self.max_steps is None) == (self.max_updates is None):
            raise ValueError("Specify exactly one of --max-steps (batches) and --max-updates")
        if (self.max_steps if self.max_steps is not None else self.max_updates) < 1:
            raise ValueError("Training budget must be positive")

    def done(self, batch_step, optimizer_step):
        return (batch_step >= self.max_steps if self.max_steps is not None
                else optimizer_step >= self.max_updates)

    def due(self, interval, batch_step, optimizer_step, updated):
        count = batch_step if self.max_steps is not None else optimizer_step
        return self.done(batch_step, optimizer_step) or (count > 0 and count % interval == 0
                                                       and (self.max_steps is not None or updated))
