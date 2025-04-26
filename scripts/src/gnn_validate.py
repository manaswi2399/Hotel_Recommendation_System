"""
Validate a trained GraphSAGE recommender on a held-out CSV.

Command to run this script
-------
python src/gnn_validate.py \
    --val_csv data/processed/Validation.csv \
    --model_dir models/
"""
import argparse, logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.data import Data

from gnn_train import GraphSAGE, build_graph, set_seeds   

def main(cfg):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=logging.INFO)
    set_seeds()

    # ---------- Load model & encoders ----------
    user_enc  = joblib.load(Path(cfg.model_dir) / "user_encoder.pkl")
    hotel_enc = joblib.load(Path(cfg.model_dir) / "hotel_encoder.pkl")

    sample_input = torch.load(Path(cfg.model_dir) / "graphsage.pth")
    emb_dim = sample_input["conv2.bias"].shape[0]        # infer output dim
    model = GraphSAGE(in_dim=len(sample_input["conv1.bias"]),
                      out_dim=emb_dim)
    model.load_state_dict(torch.load(Path(cfg.model_dir) / "graphsage.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---------- Load validation interactions ----------
    df = pd.read_csv(cfg.val_csv).dropna(subset=["author","hotel_id","rating"])
    # keep only users/hotels seen in training
    mask = df["author"].isin(user_enc.classes_) & df["hotel_id"].isin(hotel_enc.classes_)
    df = df[mask].copy()
    if df.empty:
        logging.warning("No overlapping users/hotels between train & validation!")
        return

    df["uid"] = user_enc.transform(df["author"])
    df["hid"] = hotel_enc.transform(df["hotel_id"])

    feat_cols = ["rating", "cleanliness", "service",
                 "rooms", "value", "location_rating"]
    data = build_graph(df, user_enc, hotel_enc, feat_cols).to(device)

    with torch.no_grad():
        emb = model(data.x, data.edge_index)
        users  = torch.tensor(df["uid"].values,   dtype=torch.long,  device=device)
        hotels = torch.tensor(df["hid"].values,   dtype=torch.long,  device=device) + len(user_enc.classes_)
        preds  = (emb[users] * emb[hotels]).sum(dim=1).cpu().numpy()

    true = df["rating"].values
    rmse = mean_squared_error(true, preds, squared=False)
    mae  = mean_absolute_error(true, preds)
    logging.info("Validation RMSE %.4f | MAE %.4f | n=%d", rmse, mae, len(df))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", required=True,
                   help="Path to processed Validation.csv")
    p.add_argument("--model_dir", default="models/",
                   help="Directory containing trained model & encoders")
    cfg = p.parse_args()
    main(cfg)
