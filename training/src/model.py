import os
import json
import onnx
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class CustomSentenceTransformer(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def encode_sentences(self, sentences, device="cpu", batch_size=32):
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_texts = sentences[i:i + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                for k, v in encoding.items():
                    encoding[k] = v.to(device)

                embeddings = self.forward(
                    encoding['input_ids'], 
                    encoding['attention_mask']
                )
                all_embeddings.append(embeddings.cpu())

        if not all_embeddings:
            return torch.empty(0, dtype=torch.float32).numpy()
        return torch.cat(all_embeddings, dim=0).numpy()


def export_model_to_onnx(model: CustomSentenceTransformer, onnx_path: str = 'model.onnx', max_length: int = 256):
    model.eval()
    model.to("cpu")  

    dummy_text = ["ONNX export"]
    encoding = model.tokenizer(
        dummy_text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    for k, v in encoding.items():
        encoding[k] = v.to("cpu")

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size"}
    }

    torch.onnx.export(
        model,
        (encoding['input_ids'], encoding['attention_mask']),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

def export_tokenizer_and_config(model: CustomSentenceTransformer, output_dir: str = 'models/'):
    os.makedirs(output_dir, exist_ok=True)
    model.tokenizer.save_pretrained(output_dir)
    config = model.transformer.config.to_dict()
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)