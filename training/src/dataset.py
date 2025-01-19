import ijson
import random
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset

from .utils import timer

@timer
def load_and_prepare_data(dataset_path: str, subset_size: int = None, pickle_path: str = None) -> pd.DataFrame:
    data = []
    with open(dataset_path, 'r') as f:
        for idx, obj in enumerate(ijson.items(f, 'item')):
            data.append(obj)
            if subset_size and idx + 1 >= subset_size:
                break

    df = pd.DataFrame(data)
    df.dropna(subset=['id', 'title', 'abstract', 'categories'], inplace=True)
    df['categories'] = df['categories'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    df['text'] = df['title'] + ". " + df['abstract']
    df.to_pickle(pickle_path)
    return df

@timer
def create_category_pairs(df: pd.DataFrame, num_pairs: int = 10000) -> List[Tuple[int, int, float]]:
    category_to_ids = {}
    for idx, row in df.iterrows():
        for cat in row['categories']:
            category_to_ids.setdefault(cat, []).append(idx)

    pos_pairs = []
    for cat, indices in category_to_ids.items():
        random.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            pos_pairs.append((indices[i], indices[i+1], 1.0))

    all_ids = df.index.tolist()
    neg_pairs = []
    while len(neg_pairs) < num_pairs:
        i1, i2 = random.sample(all_ids, 2)
        cat1 = set(df.loc[i1, 'categories'])
        cat2 = set(df.loc[i2, 'categories'])
        if cat1.isdisjoint(cat2):
            neg_pairs.append((i1, i2, 0.0))

    pos_pairs = pos_pairs[:num_pairs]
    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    return pairs


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pairs: List[Tuple[int, int, float]]):
        self.df = df
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, label = self.pairs[idx]
        text1 = self.df.loc[i1, 'text']
        text2 = self.df.loc[i2, 'text']
        return text1, text2, label