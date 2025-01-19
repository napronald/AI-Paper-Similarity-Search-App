import faiss
import argparse
import numpy as np
import onnxruntime
import pandas as pd
from transformers import AutoTokenizer
import warnings
import os
from colorama import init, Fore, Style
import textwrap

init(autoreset=True)

warnings.filterwarnings("ignore")

def generate_embedding(session: onnxruntime.InferenceSession, tokenizer: AutoTokenizer, query: str, max_length: int = 256) -> np.ndarray:
    encoding = tokenizer([query], padding=True, truncation=True, max_length=max_length, return_tensors="np")
    ort_inputs = {
        "input_ids": encoding['input_ids'].astype(np.int64),
        "attention_mask": encoding['attention_mask'].astype(np.int64)
    }
    return session.run(None, ort_inputs)[0]

def display_recommendations(df: pd.DataFrame, indices: np.ndarray, distances: np.ndarray, top_k: int):
    print(f"{Fore.CYAN}\nTop {top_k} Recommendations:{Style.RESET_ALL}")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1:
            continue
        paper = df.iloc[idx]
        categories = paper['categories']
        if isinstance(categories, str):
            categories = categories.split()
        
        abstract = textwrap.fill(paper['abstract'], width=185)  
        print(f"\n{Fore.YELLOW}Rank {rank}:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}ID:{Style.RESET_ALL} {paper['id']}")
        print(f"{Fore.GREEN}Title:{Style.RESET_ALL} {paper['title']}")
        print(f"{Fore.GREEN}Abstract:{Style.RESET_ALL}\n{abstract[:500]}{'...' if len(abstract) > 500 else ''}")
        print(f"{Fore.GREEN}Categories:{Style.RESET_ALL} {', '.join(categories)}")
        print(f"{Fore.GREEN}Score (Distance):{Style.RESET_ALL} {dist:.4f}")

def interactive_recommender(df: pd.DataFrame, index: faiss.Index, session: onnxruntime.InferenceSession, tokenizer: AutoTokenizer, top_k: int = 10):
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header()
        print("Type your query and press Enter to get recommendations.")
        print("Type 'exit' or 'quit' to terminate the program.\n")
        
        print(f"{Fore.MAGENTA}{'='*50}{Style.RESET_ALL}")
        query = input(f"{Fore.BLUE}Enter your query:{Style.RESET_ALL} ").strip()
        if query.lower() in ['exit', 'quit']:
            print("\nExiting the application. Goodbye!")
            break
        if not query:
            print(f"{Fore.RED}Empty query. Please enter a valid query.{Style.RESET_ALL}")
            continue
                
        print("\nProcessing your query, please wait...")
        embedding = generate_embedding(session, tokenizer, query)
        faiss.normalize_L2(embedding)
        distances, indices = index.search(embedding, top_k)
        
        display_recommendations(df, indices, distances, top_k)
        print(f"\n{Fore.MAGENTA}{'='*50}{Style.RESET_ALL}")
        input("Press Enter to enter a new query or type 'exit' to quit.")

def print_header():
    header = f"""
{Fore.CYAN}
=========================================
   AI Paper Similarity Search
=========================================
{Style.RESET_ALL}
"""
    print(header)

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
    
    print(f"{Fore.GREEN}Data and models loaded successfully!{Style.RESET_ALL}")
    interactive_recommender(df, index, session, tokenizer, top_k=args.top_k)

if __name__ == "__main__":
    main()
