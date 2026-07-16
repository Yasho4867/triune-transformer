"""
Training engine.

Eventually contains the complete training loop.
"""

class TrainingEngine:

    def __init__(self, trainer):
        self.trainer = trainer

    def train(self):
        raise NotImplementedError
