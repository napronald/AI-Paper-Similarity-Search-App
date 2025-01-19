import json
import faiss
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, precision_score, average_precision_score

from .model import CustomSentenceTransformer
from .utils import timer

@timer
def create_faiss_index(embeddings: np.ndarray, faiss_index_path: str):
    embeddings = embeddings.astype('float32') if embeddings.dtype != np.float32 else embeddings
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, faiss_index_path)
    logging.info(f"[FAISS] Flat index saved to {faiss_index_path}")
    return index

def recommend_papers(
    query: str,
    model: CustomSentenceTransformer,
    index: faiss.Index,
    df: pd.DataFrame,
    top_k: int = 10,
    device: str = "cuda"
) -> pd.DataFrame:
    query_embedding = model.encode_sentences([query], device=device, batch_size=1).astype('float32')
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    recommended_papers = df.iloc[indices[0]].copy()
    recommended_papers['score'] = distances[0]
    return recommended_papers[['id', 'title', 'abstract', 'categories', 'score']]

@timer
def evaluate_recommendations(
    model: CustomSentenceTransformer,
    index: faiss.Index,
    df: pd.DataFrame,
    test_config_path: str,
    top_k: int = 50,
    device: str = "cuda"
):
    with open(test_config_path, "r") as f:
        test_config = json.load(f)

    test_queries = [entry["query"] for entry in test_config["test_queries"]]
    test_relevant_ids = [set(entry["relevant_ids"]) for entry in test_config["test_queries"]]

    ndcg_scores, precision_scores_, recall_scores_, average_precisions_ = [], [], [], []

    for query, relevant_ids in zip(test_queries, test_relevant_ids):
        recs = recommend_papers(query, model, index, df, top_k, device=device)
        y_true = [1 if pid in relevant_ids else 0 for pid in recs['id']]
        y_score = recs['score'].tolist()

        ndcg_val = ndcg_score([y_true], [y_score], k=top_k)

        y_pred = [1 if s > 0 else 0 for s in y_score]
        prec_val = precision_score(y_true, y_pred, zero_division=0)

        relevant_retrieved = sum(y_pred)
        recall_val = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0

        ap_val = average_precision_score(y_true, y_score)

        ndcg_scores.append(ndcg_val)
        precision_scores_.append(prec_val)
        recall_scores_.append(recall_val)
        average_precisions_.append(ap_val)

        logging.info(f"Query: '{query}'")
        logging.info(f"  - NDCG@{top_k}: {ndcg_val:.4f}")
        logging.info(f"  - Precision@{top_k}: {prec_val:.4f}")
        logging.info(f"  - Recall@{top_k}:    {recall_val:.4f}")
        logging.info(f"  - AP@{top_k}: {ap_val:.4f}")

    logging.info("\n=== Final Evaluation Metrics ===")
    logging.info(f"Average NDCG@{top_k}:  {np.mean(ndcg_scores):.4f}")
    logging.info(f"Average Precision@{top_k}: {np.mean(precision_scores_):.4f}")
    logging.info(f"Average Recall@{top_k}:    {np.mean(recall_scores_):.4f}")
    logging.info(f"Average AP@{top_k}:        {np.mean(average_precisions_):.4f}")