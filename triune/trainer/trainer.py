from .engine import TrainingEngine

class Trainer:
    """
    High-level training interface.

    This will gradually replace legacy/train_llm.py.
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        train_loader,
        eval_loader,
        config,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = device

        self.engine = TrainingEngine(self)

    def fit(self):
        return self.engine.train()
