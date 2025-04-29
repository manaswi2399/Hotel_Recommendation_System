"""
Validate a trained GraphSAGE+MLPHead recommender on a held-out CSV.

Usage:
python gnn_validate.py \
  --val_csv    data/processed/Validation.csv \
  --model_dir  models/ \
  --in_dim     23
"""
import argparse, logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.data import Data

from gnn_train import GraphSAGE, MLPHead, set_seeds

def build_graph_validate(df, user_enc, hotel_enc, feat_cols, hotel_scaler, user_scaler):
    """Reconstruct the PyG Data object using saved encoders & scalers."""
    num_users  = len(user_enc.classes_)
    num_hotels = len(hotel_enc.classes_)

    # edges
    src = torch.from_numpy(df["uid"].to_numpy(dtype=np.int64))
    dst = torch.from_numpy((df["hid"] + num_users).to_numpy(dtype=np.int64))
    edge_index = torch.stack([src, dst], dim=0)

    # node features
    F_dim = len(feat_cols) + 2
    X = torch.zeros((num_users + num_hotels, F_dim), dtype=torch.float)

    # hotel features (mean per hotel)
    hf = (
        df.groupby("hid")[feat_cols]
          .mean()
          .reindex(range(num_hotels))
          .fillna(0.0)
          .values
    )
    X[num_users:, :len(feat_cols)] = torch.tensor(
        hotel_scaler.transform(hf), dtype=torch.float
    )

    # user stats (mean,rating count)
    us = (
        df.groupby("uid")["rating"]
          .agg(["mean","count"])
          .reindex(range(num_users))
          .fillna(0.0)
          .values
    )
    X[:num_users, -2:] = torch.tensor(
        user_scaler.transform(us), dtype=torch.float
    )

    return Data(x=X, edge_index=edge_index)

def main(cfg):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=logging.INFO)
    set_seeds()

    # 1) Load encoders & scalers
    model_dir = Path(cfg.model_dir)
    user_enc     = joblib.load(model_dir / "user_encoder.pkl")
    hotel_enc    = joblib.load(model_dir / "hotel_encoder.pkl")
    hotel_scaler = joblib.load(model_dir / "hotel_scaler.pkl")
    user_scaler  = joblib.load(model_dir / "user_scaler.pkl")

    # 2) Instantiate models with same dims used in training
    gnn = GraphSAGE(in_dim=cfg.in_dim, hid_dim=128, out_dim=64)
    head = MLPHead(embed_dim=64)

    # 3) Load weights
    gnn.load_state_dict(torch.load(model_dir / "graphsage.pth"))
    head.load_state_dict(torch.load(model_dir / "mlp_head.pth"))
    gnn.eval(); head.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn, head = gnn.to(device), head.to(device)

    # 4) Load validation data
    df = pd.read_csv(cfg.val_csv).dropna(subset=["author","hotel_id","rating"])
    mask = df["author"].isin(user_enc.classes_) & df["hotel_id"].isin(hotel_enc.classes_)
    df = df[mask].copy()
    if df.empty:
        logging.warning("No overlapping users/hotels; exiting.")
        return

    # 5) Encode IDs for edges
    df["uid"] = user_enc.transform(df["author"])
    df["hid"] = hotel_enc.transform(df["hotel_id"])

    # 6) Rebuild graph with metadata features
    feat_cols = [
        "cleanliness","service","rooms","value","location_rating",
        "sentiment_score",
        "has_room_quality","has_location","has_staff_service",
        "has_cleanliness","has_amenities","has_value_for_money",
        "has_food_and_restaurant","has_noise_level","has_comfort",
        "is_cleanliness_missing","is_service_missing",
        "is_rooms_missing","is_sleep_quality_missing",
        "is_value_missing","is_location_rating_missing"
    ]
    data = build_graph_validate(df, user_enc, hotel_enc,
                                feat_cols, hotel_scaler, user_scaler).to(device)

    # 7) Forward pass & predictions
    with torch.no_grad():
        emb    = gnn(data.x, data.edge_index)  # (N_nodes, 64)
        users  = torch.tensor(df["uid"].values, device=device)
        hotels = torch.tensor(df["hid"].values + len(user_enc.classes_),
                              device=device)
        raw = head(emb[users], emb[hotels])
        preds  = (torch.sigmoid(raw) * 5.0).cpu().numpy()

    # 8) Compute metrics
    true = df["rating"].values
    rmse = mean_squared_error(true, preds)
    mae  = mean_absolute_error(true, preds)
    logging.info("Sample true: %s", true[:5])
    logging.info("Sample pred: %s", preds[:5])
    logging.info("Validation RMSE %.4f | MAE %.4f | n=%d", rmse, mae, len(df))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv",   required=True)
    p.add_argument("--model_dir", default="models/")
    p.add_argument("--in_dim",    type=int, required=True,
                   help="Number of node features used during training")
    cfg = p.parse_args()
    main(cfg)