import faiss
import argparse
import numpy as np
import onnxruntime
import pandas as pd
from transformers import AutoTokenizer

def generate_embedding(session: onnxruntime.InferenceSession, tokenizer: AutoTokenizer, query: str, max_length: int = 256) -> np.ndarray:
    encoding = tokenizer([query], padding=True, truncation=True, max_length=max_length, return_tensors="np")
    ort_inputs = {
        "input_ids": encoding['input_ids'].astype(np.int64),
        "attention_mask": encoding['attention_mask'].astype(np.int64)
    }
    return session.run(None, ort_inputs)[0]

def display_recommendations(df: pd.DataFrame, indices: np.ndarray, distances: np.ndarray, top_k: int):
    print(f"\nTop {top_k} Recommendations:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        paper = df.iloc[idx]
        categories = paper['categories']
        if isinstance(categories, str):
            categories = categories.split()  
        print(f"\nRank {rank}:")
        print(f"ID: {paper['id']}")
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract'][:200]}...")
        print(f"Categories: {', '.join(categories)}")
        print(f"Score (Distance): {dist:.4f}")

def interactive_recommender(df: pd.DataFrame, index: faiss.Index, session: onnxruntime.InferenceSession, tokenizer: AutoTokenizer, top_k: int =10):
    print("\n=== AI Paper Similarity Search ===")
    print("Type your query and press Enter to get recommendations.")
    print("Type 'exit' or 'quit' to terminate the program.\n")
    
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Exiting")
            break
        if not query:
            print("Empty query. Please enter a valid query.")
            continue
                
        embedding = generate_embedding(session, tokenizer, query)
        faiss.normalize_L2(embedding)
        distances, indices = index.search(embedding, top_k)
        
        display_recommendations(df, indices, distances, top_k)

def main():
    parser = argparse.ArgumentParser(description="AI Paper Similarity Search")
    parser.add_argument("--pickle", type=str, default="papers.pkl", help="Path to the dataframe pickle file")
    parser.add_argument("--index_path", type=str, default="index.bin", help="Path to the FAISS index file")
    parser.add_argument("--model_path", type=str, default="model.onnx", help="Path to the ONNX model file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top recommendations to display")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Name of the transformer model used for tokenization")
    args = parser.parse_args()

    df = pd.read_pickle(args.pickle)
    index = faiss.read_index(args.index_path)
    session = onnxruntime.InferenceSession(args.model_path, providers=['CPUExecutionProvider'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    interactive_recommender(df, index, session, tokenizer, top_k=args.top_k)


if __name__ == "__main__":
    main()