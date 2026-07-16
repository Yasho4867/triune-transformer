
import torch
from torch.profiler import profile, ProfilerActivity
from model import TriuneTransformer
from tokenizers import Tokenizer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer to get vocab size
tokenizer = Tokenizer.from_file("triune_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
pad_token_id = tokenizer.token_to_id("[PAD]")

# Instantiate model (same architecture as training)
model = TriuneTransformer(vocab_size=vocab_size, hidden_dim=1536, num_layers=24)
model = model.to(device).bfloat16()
model.train()

# Load the latest checkpoint if available
checkpoint_path = "checkpoints_full/latest.pt"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print("Loaded checkpoint")
else:
    print("No checkpoint found – using fresh model")

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

# Dummy data: batch_size=4, seq_len=256 (matches your training config)
batch_size = 4
seq_len = 256

print("Warming up (5 steps to trigger compilation)...")
for _ in range(5):
    model.zero_grad(set_to_none=True)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = model(x, force_depth=2)
        loss = loss_fn(logits.contiguous().view(-1, vocab_size), y.contiguous().view(-1))
    loss.backward()

print("Running profile (10 steps) with CPU activities only (to avoid CUPTI errors)...")
with profile(
    activities=[ProfilerActivity.CPU],  # CPU only – avoids CUPTI issues
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(10):
        model.zero_grad(set_to_none=True)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x, force_depth=2)
            loss = loss_fn(logits.contiguous().view(-1, vocab_size), y.contiguous().view(-1))
        loss.backward()

# Print and save the table
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Save to file
with open("profiler_output.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
print("\n✅ Profiler output saved to profiler_output.txt")
print("   Please open that file and paste the contents here.")
