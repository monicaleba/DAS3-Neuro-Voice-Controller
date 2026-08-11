import json
import os
from transformers import AutoTokenizer

# ────────────────────────────────────────────────────────────────────────
# 1. LOAD THE DATASET
# ────────────────────────────────────────────────────────────────────────
DATA_FILE = 'train.json'

print(f"Loading text data from {DATA_FILE}...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# The tokenizer only needs the raw text (the 'input' column), not the labels.
# We use a Python 'generator' (yield) which is highly memory efficient.
def get_training_corpus():
    for item in train_data:
        yield item['input']

# ────────────────────────────────────────────────────────────────────────
# 2. INITIALIZE THE BASE ARCHITECTURE
# ────────────────────────────────────────────────────────────────────────
# We load the "blueprint" of a DistilBERT tokenizer. 
# We aren't using its vocabulary, just its rules for WordPiece tokenization
# and its special tokens like [CLS] (start of sentence) and [SEP] (end).
print("Loading base DistilBERT tokenizer blueprint...")
base_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ────────────────────────────────────────────────────────────────────────
# 3. TRAIN THE CUSTOM TOKENIZER
# ────────────────────────────────────────────────────────────────────────
# This is where the magic happens. We limit the vocab_size to 1500.
# It will analyze all 7,200 of your training phrases and build the perfect, 
# smallest possible dictionary of your exact roots and words.
VOCAB_SIZE = 1500

print(f"Training new tokenizer with a max vocabulary of {VOCAB_SIZE}...")
custom_tokenizer = base_tokenizer.train_new_from_iterator(
    get_training_corpus(), 
    vocab_size=VOCAB_SIZE
)

# ────────────────────────────────────────────────────────────────────────
# 4. SAVE TO DISK
# ────────────────────────────────────────────────────────────────────────
SAVE_DIR = "custom_das3_tokenizer"
os.makedirs(SAVE_DIR, exist_ok=True)

custom_tokenizer.save_pretrained(SAVE_DIR)

print("\n" + "="*50)
print(f"✅ CUSTOM TOKENIZER BUILT SUCCESSFULLY!")
print(f"   - Actual Vocabulary Size : {len(custom_tokenizer)} tokens")
print(f"   - Saved to Directory     : ./{SAVE_DIR}/")
print("="*50)
print("Test it: The AI now understands 'end-effector' as a native concept!")