import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import CHECKPOINT_DIR, HIDDEN_DIM, NUM_LAYERS
from model import TriuneTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_DIR + "/best.pt")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
parser.add_argument("--max_new_tokens", type=int, default=100)
args = parser.parse_args()

if args.temperature <= 0:
    parser.error("--temperature must be greater than zero")
if args.repetition_penalty < 0:
    parser.error("--repetition_penalty must be non-negative")
if args.max_new_tokens <= 0:
    parser.error("--max_new_tokens must be greater than zero")

TOKENIZER_PATH = "triune_tokenizer.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
pad_token_id = tokenizer.token_to_id("[PAD]")
eos_token_id = tokenizer.token_to_id("[SEP]")
if pad_token_id is None or eos_token_id is None:
    raise ValueError("Tokenizer must define [PAD] and [SEP] special tokens")

print("Loading model...")
checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
ckpt_cfg = checkpoint.get("config", {})
state_dict = checkpoint["model_state"]
if any(key.startswith("_orig_mod.") for key in state_dict):
    state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

# Older checkpoints did not record this setting.  Infer it from the expert
# parameter names so their state dictionaries remain loadable.
if "use_fp4" in ckpt_cfg:
    use_fp4 = ckpt_cfg["use_fp4"]
else:
    use_fp4 = any(".ffn.experts." in key and ".linear.weight" in key for key in state_dict)

model = TriuneTransformer(
    vocab_size=ckpt_cfg.get("vocab_size", vocab_size),
    hidden_dim=ckpt_cfg.get("hidden_dim", HIDDEN_DIM),
    num_layers=ckpt_cfg.get("num_layers", NUM_LAYERS),
    use_fp4=use_fp4,
)
model.load_state_dict(state_dict)
model = model.to(DEVICE).bfloat16().eval()
print(f"Model loaded. Best eval loss: {checkpoint.get('best_eval_loss', 'unknown')}")

@torch.inference_mode()
def generate_response(prompt, force_depth=None):
    formatted = f"User: {prompt}\nAssistant:"
    ids = torch.tensor(tokenizer.encode(formatted).ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = []
    seen = {}

    for _ in range(args.max_new_tokens):
        logits, _ = model(ids, force_depth=force_depth)
        next_logits = logits[0, -1].float() / args.temperature
        for token, count in seen.items():
            next_logits[token] -= args.repetition_penalty * count
        next_logits[pad_token_id] = -torch.inf

        next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1).item()
        if next_id == eos_token_id:
            break
        generated.append(next_id)
        seen[next_id] = seen.get(next_id, 0) + 1
        ids = torch.cat((ids, torch.tensor([[next_id]], device=DEVICE)), dim=1)

    return tokenizer.decode(generated)

print("Triune Transformer chat mode. Commands: reflex / limbic / cortex / auto; exit to quit.")
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break

    force_depth = None
    for command, depth in (("reflex ", 0), ("limbic ", 1), ("cortex ", 2)):
        if user_input.lower().startswith(command):
            force_depth = depth
            user_input = user_input[len(command):]
            break
    if user_input.lower().startswith("auto "):
        user_input = user_input[len("auto "):]
    print(f"Assistant: {generate_response(user_input, force_depth)}")
