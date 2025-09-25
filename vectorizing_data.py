from transfomers import AutoTokenizer, AutomodelForSequenceClassification, TextClassificationPipeline
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch

#Change this path to which company you are analyzing
CSV_PATH = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_qna_turns.csv'
TEXT_COL = 10 
MODEL_ID = 'ProsusAI/finbert-tone', have to choose which one

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 256 #Could change to 512
STRIDE = 32 #Lets see what optimizes?

df = pd.read_csv(CSV_PATH)

texts = df[TEXT_COL].astype(str).tolist()

tok = AutoTokenizer.from_pretrained(MODEL_ID)

#FinBERT classifier for sentiment Probablities
#This has a small classification head on top of BERT
clf = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
clf = clf.to(DEVICE).eval()

#FinBERT base model for embeddings
base = AutoModel.from_pretrained(MODEL_ID)
base = base.to(DEVICE).eval()

def chunk_encode(text, max_length = MAX_LEN, stride = STRIDE):

    # Step 1 Is to Tokenize without truncation because we want the full token sequence
    enc = tok(text, return_tensors = "pt", truncation = FALSE)

    # Step 2 Is to unpack the token ID's and attention mask
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    # Step 3 is to slide a window across the sequence, why?
    chunks = []
    start = 0 
    while start < len(input_ids):
        end = min(start + MAX_LEN, len(input_ids))

        #Slicing the window 
        ids_slice = input_ids[start:end]
        am_slice = attention_mask[start:end]

        #Add batch dimension as model expects the tensors to have a batch dimension s.t [B, T]
        "input_ids": ids_slice.unsqueeze(0),
        "attention_mask": am_slice.unsqueeze(0)

        chunks.append({
            "input_ids": ids_slice.unsqueeze(0),
            "attention_mask": am_slice.unsqueeze(0)
        })

        if end == len(input_ids):
            break 

        #This part moves the window forward overlapping by 'stride'
        start = end - stride
        if start < 0:
            start = 0

    return chunks

def finbert_probs(text):
    chunks = chunk_encode(text)

    all_logits = []
    for ch in chunks:
        ch = {k: v.to(DEVICE) for k, v in ch.items()} #THis code is copied but moves the tensors from GPU/CPOU
        out = clf(**ch)
        all_logits.append(out.logits)

    #Stacking into [num_chunks, 3] and average across chunks
    logits = torch.cat(all_logits, dim = 0).mean(dim = 0)
    probs = torch.softmax(logits, dim = -1),cpu().numpy()

    return probs 

def finbert_embedding(text, pool = "cls"):
    """
    Compute a 768-d embedding for the text by:
      - Running each chunk through the base BERT
      - Pooling token representations per chunk (CLS or mean)
      - Averaging chunk vectors to a single vector

    pool = "cls"  -> use the [CLS] token (position 0) from last hidden layer
    pool = "mean" -> attention-mask-weighted mean over all tokens
    """

    chunks = chunk_encode(text)
    vecs = []
