!pip -q install transformers torch tqdm pandas pyarrow

# Arrays & math
import numpy as np

# DataFrames for handling CSVs
import pandas as pd

# Progress bar (nice to have)(AI suggestion)
from tqdm import tqdm

# Deep learning backend
import torch

# Hugging Face models & tokenizers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# OS for environment variables
import os

# Disable Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# File path to CSV
CSV_PATH = r"C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\run_3_amd_qna_turns.csv"
df = pd.read_csv(CSV_PATH)
TEXT_COL = "Text"
texts = df[TEXT_COL].astype(str).fillna("").tolist()

# Load FinBERT model
MODEL_ID = "ProsusAI/finbert"

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking parameters
MAX_LEN, STRIDE = 256, 32

tok  = AutoTokenizer.from_pretrained(MODEL_ID)

# This loads the FinBERT Tokenizer:
# 1. Splits raw text into subword tokens (BERT uses WordPiece, e.g. expectations → expect + ##ations)
# 2. Maps tokens to integer IDs in FinBERT vocab
# 3. Adds special tokens like CLS (start) and SEP (end) → [CLS], The, company, beat, expect, ##ations, [SEP]
# Important: the tokenizer must match the model as they differ

clf  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
# clf is classifier, loads FinBERT with its classification head attached.
# This head is a small linear layer mapping BERT's CLS output to 3 sentiment classes (positive, negative, neutral)

base = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
# Loads only the encoder part of FinBERT, it outputs hidden states for every token (each 768-dimensional)

id2label = clf.config.id2label
print("Device:", DEVICE, "| Labels:", id2label)

# Now lets start making the sliding window to compensate for small max token length
def chunk_encode(text, max_len = MAX_LEN, stride = STRIDE):
    enc = tok(text, return_tensors = "pt", truncation = False)
    ids  = enc["input_ids"][0]        # [total_tokens]
    mask = enc["attention_mask"][0]   # [total_tokens]
    # We remove the batch dimension by indexing [0]

    chunks = []
    start = 0 # Left index of the current window on the token axis 

    while start < len(ids):
        end = min(start + max_len, len(ids))
        
        # Take a 1-d slice of token IDs and the corresponding attention mask
        # This makes the length T = end - start (≤ max_len)
    
        ids_slice = ids[start:end]
        mask_slice = mask[start:end]
    
        # Unsqueeze as model expects a batch dimension.
        # After this each appended dict has exactly the keys the model's forward expects
        # We are creating a dictionary which we can then loop through chunks of the text  
        chunks.append({
            "input_ids": ids_slice.unsqueeze(0),
            "attention_mask": mask_slice.unsqueeze(0),
        })

        # Now we decide to stop or move the sliding window
        if end == len(ids):
            break
        start = end - stride 

    return chunks

@torch.no_grad()
def finbert_probs(text):
    logits_list = []
    for ch in chunk_encode(text):
        out = clf(**{k: v.to(DEVICE) for k, v in ch.items()})
        # After passing it through FinBERT we get out.logits with shape [1, 3]
        # These are our three sentiment classes: positive, negative, neutral
        logits_list.append(out.logits)
        # Save each chunk's logits 
    
    # Now we return one logit vector for the whole text by averaging
    L = torch.cat(logits_list, dim = 0).mean(dim = 0) 
    return torch.softmax(L, dim = -1).cpu().numpy()

@torch.no_grad()
def embed_one_chunk(ch, pool = "mean"):
    ch = {k: v.to(DEVICE) for k, v in ch.items()}
    base.eval()
    out = base(**ch) # This gives the last hidden state [1, T, 768]
    H = out.last_hidden_state
    print("H:", H.shape)
    
    if pool == "cls":
        v = H[:, 0, :]
        # We take [CLS] as BERT is designed so that CLS accumulates information from the whole sequence via self-attention
        print("CLS v:", v.shape)   
    else:
        mask = ch["attention_mask"].unsqueeze(-1) 
        # [1, T, 1], we unsqueeze from [1, T] so we can multiply
        print("mask:", mask.shape)
        summed = (H * mask).sum(dim = 1) 
        # H * mask zeroes out the embeddings of padding tokens
        # [1, T, 768] * [1, T, 1] = [1, T, 768], and then .sum(dim = 1) sums across T tokens
        # This leaves [1, 768] because it’s elementwise multiplication then reduction
        counts = mask.sum(dim = 1).clamp(min = 1e-9) # [1, 1]
        # This counts how many real tokens each example has (batch-wise)
        v = summed / counts 
        # [1, 768] / [1, 1] = [1, 768]
        # Mean pooling gives a more global embedding that considers all tokens (not just CLS) 
        print("summed:", summed.shape, "counts:", counts.shape, "v:", v.shape)
    print("final v:", v.squeeze(0).shape)
    return v.squeeze(0).cpu().numpy()

@torch.no_grad()
def finbert_embedding(text, pool = "cls"):
    vecs = []
    # Wrap chunk_encode with tqdm for progress bar
    for ch in tqdm(chunk_encode(text), desc = "Embedding chunks"):
        v = embed_one_chunk(ch, pool = pool)   # [768]
        vecs.append(v)
        
    # Each chunk in list chunk_encode(text) will produce one vector [768] via embed_one_chunk
    return np.stack(vecs, axis = 0).mean(axis = 0) 
    # Turns list of vectors into a 2D array with [num_chunks, 768]
    # We stack then mean as we can then get the average across all chunks

prob_list, emb_list = [], []
for t in tqdm(texts, desc = "Embedding with FinBERT"):
    try:
        p = finbert_probs(t) # [3], this is the averaged over the chunks class probabilities
        e = finbert_embedding(t, pool = "cls") # [768]
    except Exception:
        # This is defensive code, catches unexpected failure
        # How? it produces placeholders:
        # NaNs with shape [3], zeros with shape [768]
        p = np.array([np.nan, np.nan, np.nan])
        e = np.zeros(768, dtype = np.float32)
        
    prob_list.append(p)
    emb_list.append(e)

print("len(texts) =", len(texts))
print("len(prob_list) =", len(prob_list))
print("len(emb_list) =", len(emb_list))

if len(prob_list) > 0:
    probs = np.vstack(prob_list)
    embs  = np.vstack(emb_list)
    print("probs shape:", probs.shape)  # (n,3)
    print("embs shape:", embs.shape)    # (n,768)
else:
    print("No arrays collected!")

probs_cols = [f"finbert_prob_{id2label[i].lower()}" for i in range(len(id2label))]
# Creates column names in the model’s label order, just to follow the order the model uses
probs_df = pd.DataFrame(probs, columns = probs_cols)
# Wrapping probs [n, 3] array into a DataFrame with readable column names 
emb_cols = [f"finbert_emb_{i}" for i in range(embs.shape[1])]
emb_df = pd.DataFrame(embs, columns = emb_cols)
# Same logic for embeddings

features = pd.concat([df.reset_index(drop=True), probs_df, emb_df], axis = 1)
# Horizontally concatenate original data + the new features 
# After concat: features.shape == (n_rows, original_cols + 3 + 768)
# Tried without reset but this ends up with misaligned rows 

assert len(features) == len(df)
print(features.columns[:10])

# Add predicted label + confidence
pred_idx = probs.argmax(axis=1)
pred_lab = np.vectorize(lambda i: id2label[i])(pred_idx)

conf_max = probs.max(axis=1)

features.insert(features.columns.get_loc(probs_cols[-1]) + 1, "finbert_pred_label", pred_lab)
features.insert(features.columns.get_loc("finbert_pred_label") + 1, "finbert_confidence", conf_max)

OUT_PATH = "finbert_embeddings_plus_probs.parquet"
features.to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print("Shape:", features.shape)
