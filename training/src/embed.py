import numpy as np
import pandas as pd
from tqdm import tqdm

from .model import CustomSentenceTransformer
from .utils import timer

@timer
def generate_embeddings(
    df: pd.DataFrame,
    model: CustomSentenceTransformer,
    device: str = "cuda",
    embed_path: str = "embeddings.npy",
    batch_size: int = 128
) -> np.ndarray:
    texts = df['text'].tolist()
    num_samples = len(texts)

    model.to(device).eval()

    sample_embedding = model.encode_sentences(texts[:2], device=device, batch_size=2)
    embedding_dim = sample_embedding.shape[1] if sample_embedding.size else 384
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i : i + batch_size]
        emb_batch = model.encode_sentences(batch_texts, device=device, batch_size=batch_size)
        embeddings[i : i + len(emb_batch)] = emb_batch.astype("float32")

    np.save(embed_path, embeddings)
    return embeddings