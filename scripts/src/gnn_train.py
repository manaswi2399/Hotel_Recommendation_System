#!/usr/bin/env python3
"""
Train a GraphSAGE recommender (user-hotel bipartite graph).

Commad to run this script
-------
python src/gnn_train.py \
    --train_csv data/processed/Train.csv \
    --model_dir models/ \
    --epochs 50
"""
import argparse, logging, os, random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

"""
Helper functions
"""
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_graph(df: pd.DataFrame,
                user_enc: LabelEncoder,
                hotel_enc: LabelEncoder,
                feat_cols: list[str]) -> Data:
    """Return PyG Data object with user+hotel node features and edges."""
    num_users  = len(user_enc.classes_)
    num_hotels = len(hotel_enc.classes_)

    # Edge index (2 × E)
    edge_index = torch.tensor(
        [df["uid"].to_numpy(),
         df["hid"].to_numpy() + num_users],   # offset hotels
        dtype=torch.long
    )

    # Node-feature matrix 
    F_dim = len(feat_cols) + 2        # +2 user features (mean_rating, n_reviews)
    N_nodes = num_users + num_hotels
    X = torch.zeros((N_nodes, F_dim), dtype=torch.float)

    # Hotel features 
    X[num_users:, :len(feat_cols)] = torch.tensor(
        df.groupby("hid")[feat_cols].mean()
          .reindex(range(num_hotels))            
          .fillna(0.0).values,
        dtype=torch.float
    )

    # User features: mean rating & review count
    user_stats = df.groupby("uid")["rating"].agg(["mean", "count"])
    X[:num_users, -2:] = torch.tensor(
        user_stats.reindex(range(num_users)).fillna(0.0).values,
        dtype=torch.float
    )

    return Data(x=X, edge_index=edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x                                 # (N_nodes, out_dim)

# Training
def main(cfg):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=logging.INFO)
    set_seeds()

    df = pd.read_csv(cfg.train_csv)
    logging.info("Loaded %d interactions", len(df))

    # Encode IDs
    user_enc  = LabelEncoder().fit(df["author"])
    hotel_enc = LabelEncoder().fit(df["hotel_id"])
    df["uid"] = user_enc.transform(df["author"])
    df["hid"] = hotel_enc.transform(df["hotel_id"])

    # Build graph
    feat_cols = ["rating", "cleanliness", "service",
                "rooms", "value", "location_rating",
                "sentiment_score",
                "has_room_quality", "has_location", "has_staff_service",
                "has_cleanliness", "has_amenities", "has_value_for_money",
                "has_food_and_restaurant", "has_noise_level", "has_comfort",
                "is_cleanliness_missing", "is_service_missing",
                "is_rooms_missing", "is_sleep_quality_missing",
                "is_value_missing", "is_location_rating_missing"]

    data = build_graph(df, user_enc, hotel_enc, feat_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GraphSAGE(in_dim=data.num_node_features).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    targets = torch.tensor(df["rating"].values, dtype=torch.float, device=device)
    users   = torch.tensor(df["uid"].values,    dtype=torch.long,  device=device)
    hotels  = torch.tensor(df["hid"].values,    dtype=torch.long,  device=device) + len(user_enc.classes_)

    for epoch in range(1, cfg.epochs + 1):
        optimiser.zero_grad()
        emb = model(data.x, data.edge_index)           
        preds = (emb[users] * emb[hotels]).sum(dim=1)   
        loss  = F.mse_loss(preds, targets)
        loss.backward()
        optimiser.step()

        if epoch % 10 == 0 or epoch == cfg.epochs:
            logging.info("Epoch %3d | train MSE %.4f", epoch, loss.item())

    # Save artifacts
    Path(cfg.model_dir).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(),      Path(cfg.model_dir) / "graphsage.pth")
    joblib.dump(user_enc,               Path(cfg.model_dir) / "user_encoder.pkl")
    joblib.dump(hotel_enc,              Path(cfg.model_dir) / "hotel_encoder.pkl")
    logging.info("Training complete, model saved to %s", cfg.model_dir)

# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True,
                   help="Path to processed Train.csv")
    p.add_argument("--model_dir", default="models/",
                   help="Directory to save model & encoders")
    p.add_argument("--epochs", type=int, default=50)
    cfg = p.parse_args()
    main(cfg)
