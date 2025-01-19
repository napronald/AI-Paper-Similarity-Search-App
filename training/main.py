import argparse
import logging
import torch
import warnings 

from src.dataset import load_and_prepare_data
from src.finetune import fine_tune_model
from src.embed import generate_embeddings
from src.model import CustomSentenceTransformer, export_model_to_onnx, export_tokenizer_and_config
from src.faiss_index import create_faiss_index, evaluate_recommendations
from src.utils import timer

warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=FutureWarning)  

@timer
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[DEVICE] Using device: {device}")

    df = load_and_prepare_data(args.dataset_path, subset_size=args.dataset_size, pickle_path=args.pickle_path)
    logging.info(f"[DATA] DataFrame saved {args.dataset_size} datapoints to {args.pickle_path}")

    if args.fine_tune:
        logging.info(f"[FINE TUNE] Using {args.fine_tune_size} datapoints for fine tuning.")
        model = fine_tune_model(
            df=df,
            fine_tune_size=500000,
            num_pairs=50000,
            epochs=15,
            batch_size=512,
            margin=1.0,
            device=device
        )
    else:
        logging.info("[FINE TUNE] Skipping fine-tuning; using base model directly.")
        model = CustomSentenceTransformer().to(device)

    embeddings = generate_embeddings(df, model, device=device, embed_path=args.embed_path, batch_size=512)
    logging.info(f"[DATA] Embeddings saved to {args.embed_path}")

    index = create_faiss_index(embeddings, args.index_path)
    logging.info("[FAISS] Index creation completed.")

    export_model_to_onnx(model)
    logging.info(f"[EXPORT] ONNX model exported")

    export_tokenizer_and_config(model)
    logging.info("[EXPORT] Tokenizer exported.")

    evaluate_recommendations(
        model=model.to(device),
        index=index,
        df=df,
        test_config_path=args.test_json,
        top_k=50,
        device=device
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler("output.log"), logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="AI Paper Similarity Search")
    parser.add_argument("--dataset_path", type=str, default="processed_data.json",
                        help="Path to the processed dataset JSON file")
    parser.add_argument("--dataset_size", type=int, default=1940800,
                        help="Number of samples to use (Default: entire dataset)")
    parser.add_argument("--pickle_path", type=str, default="papers.pkl",
                        help="Path to save/load the dataframe pickle file")
    parser.add_argument("--embed_path", type=str, default="embeddings.npy",
                        help="Path to save/load the embeddings (NumPy array) file")
    parser.add_argument("--index_path", type=str, default="index.bin",
                        help="Path to save/load the FAISS index")
    parser.add_argument("--test_json", type=str, default="test_dataset.json",
                        help="Path to the JSON file containing test query and relevant IDs")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Whether to perform fine-tuning")
    args = parser.parse_args()
    main(args)