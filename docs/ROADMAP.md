# Refactor status

The callable-framework migration is complete.

- Runtime configuration is isolated from research defaults.
- Data, tokenizer, optimizer, precision, checkpoint, evaluation, and training
  execution are reusable modules.
- CLI scripts are thin entry points and are safe to import.
- Legacy scripts, empty placeholders, stale profiling output, and unrelated
  configuration artifacts were removed.

Operational documentation lives in [FRAMEWORK.md](FRAMEWORK.md).
