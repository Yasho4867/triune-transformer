class TrainingEngine:
    """
    Owns the actual training loop.

    Initially delegates to the legacy implementation.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def train(self):
        raise NotImplementedError(
            "Training loop has not yet been migrated from legacy/train_llm.py"
        )
