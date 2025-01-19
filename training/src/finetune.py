import torch
import logging
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import PairDataset, create_category_pairs
from .model import CustomSentenceTransformer
from .utils import timer


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, label):
        euclidean_distance = torch.sqrt(torch.sum((emb_a - emb_b) ** 2, dim=1) + 1e-8)

        loss_sim = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_dis = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(loss_sim + loss_dis)
        return loss

@timer
def fine_tune_model(
    df: pd.DataFrame,
    fine_tune_size: int,
    num_pairs: int = 100000,
    epochs: int = 10,
    batch_size: int = 64,
    margin: float = 1.0,
    device: str = "cuda"
) -> CustomSentenceTransformer:
    finetune_df = df.head(fine_tune_size).copy()
    pairs = create_category_pairs(df, num_pairs=num_pairs)
    dataset = PairDataset(finetune_df, pairs)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    model = CustomSentenceTransformer().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler(enabled=(device == "cuda"))
    criterion = ContrastiveLoss(margin=margin)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for text1, text2, label in progress_bar:
            encoding1 = model.tokenizer(
                list(text1),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            encoding2 = model.tokenizer(
                list(text2),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            for k, v in encoding1.items():
                encoding1[k] = v.to(device)
            for k, v in encoding2.items():
                encoding2[k] = v.to(device)

            label_tensor = label.float().to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device == "cuda")):
                emb1 = model(encoding1['input_ids'], encoding1['attention_mask'])
                emb2 = model(encoding2['input_ids'], encoding2['attention_mask'])
                loss = criterion(emb1, emb2, label_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logging.info(f"[Fine-Tune] Epoch {epoch+1}/{epochs} | Avg Contrastive Loss: {avg_loss:.4f}")

    return model