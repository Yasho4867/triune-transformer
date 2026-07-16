"""
Triune Trainer

This will gradually absorb the legacy train_llm.py.
"""

class Trainer:

    def __init__(self, config):
        self.config = config

    def fit(self):
        raise NotImplementedError("Training engine not migrated yet.")
