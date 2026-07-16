from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from datasets import load_dataset
from tqdm import tqdm

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TARGET_CHARS = 5_000_000_000

print(f"Building Triune tokenizer (vocab size: {VOCAB_SIZE})")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
)

stats = {"chunks": 0, "chars": 0}

def text_iterator():
    """Yield documents directly to the trainer without retaining the corpus in RAM."""
    for sample in tqdm(dataset, desc="Streaming text"):
        text = sample.get("text", "")
        if not text.strip():
            continue
        stats["chunks"] += 1
        stats["chars"] += len(text)
        yield text
        if stats["chars"] >= TARGET_CHARS:
            return

print("Training tokenizer...")
tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
tokenizer.save("triune_tokenizer.json")
print(f"Tokenizer saved after {stats['chunks']:,} chunks / {stats['chars']:,} characters")

test_text = "The capital of France is Paris."
encoded = tokenizer.encode(test_text)
print(f"Test encoding: {encoded.tokens}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")
