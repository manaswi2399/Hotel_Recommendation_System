"""
Train a regularized GraphSAGE recommender with fixed hyper-parameters.

Usage (no tunable flags):
    python src/gnn_train.py \
        --train_csv data/processed/Train.csv \
        --model_dir models/
"""
import argparse, logging, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# HYPER-PARAMETERS 
EPOCHS        = 100
LR            = 2e-4
WEIGHT_DECAY  = 1e-6
DROPOUT       = 0.1
HID_DIM       = 128
OUT_DIM       = 64
HEAD_HID_DIM  = 128
STEP_SIZE     = 50
GAMMA         = 0.8
SEED          = 42
LOG_INTERVAL  = 5


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_graph(df, feat_cols):
    """Build bipartite user–hotel graph and z-scaled node features."""
    # encode users/hotels
    uenc = LabelEncoder().fit(df["author"])
    henc = LabelEncoder().fit(df["hotel_id"])
    df["uid"] = uenc.transform(df["author"])
    df["hid"] = henc.transform(df["hotel_id"])
    U, H = len(uenc.classes_), len(henc.classes_)

    # edges
    src = torch.from_numpy(df["uid"].to_numpy(dtype=np.int64))
    dst = torch.from_numpy((df["hid"] + U).to_numpy(dtype=np.int64))
    edge_index = torch.stack([src, dst], dim=0)

    # node features
    F_dim = len(feat_cols) + 2
    X = torch.zeros((U + H, F_dim), dtype=torch.float)

    # hotel features
    hf = (df.groupby("hid")[feat_cols]
            .mean()
            .reindex(range(H))
            .fillna(0.0)
            .values)
    hotel_scaler = StandardScaler().fit(hf)
    X[U:, :len(feat_cols)] = torch.tensor(hotel_scaler.transform(hf))

    # user stats
    us = (df.groupby("uid")["rating"]
            .agg(["mean","count"])
            .reindex(range(U))
            .fillna(0.0)
            .values)
    user_scaler = StandardScaler().fit(us)
    X[:U, -2:] = torch.tensor(user_scaler.transform(us))

    return Data(x=X, edge_index=edge_index), uenc, henc, hotel_scaler, user_scaler

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=HID_DIM, out_dim=OUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv1  = SAGEConv(in_dim, hid_dim)
        self.conv2  = SAGEConv(hid_dim, out_dim)
        self.dropout= dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class MLPHead(torch.nn.Module):
    def __init__(self, embed_dim, hid_dim=HEAD_HID_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hid_dim, 1)
        )
    def forward(self, u, h):
        return self.net(torch.cat([u,h], dim=1)).squeeze(1)

def main(cfg):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    set_seeds()

    # 1) load & log
    df = pd.read_csv(cfg.train_csv)
    logging.info("Loaded %d interactions", len(df))

    # 2) graph & encoders & scalers
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
    data, uenc, henc, hscaler, uscaler = build_graph(df, feat_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)

    # 3) models
    model = GraphSAGE(in_dim=data.num_node_features).to(device)
    head  = MLPHead(embed_dim=OUT_DIM).to(device)

    # 4) optimizer & scheduler
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    # 5) prepare training indices
    users  = torch.tensor(df["uid"].values, device=device)
    hotels = torch.tensor(df["hid"].values + len(uenc.classes_), device=device)
    target = torch.tensor(df["rating"].values, dtype=torch.float, device=device)

    # 6) train loop
    model.train(); head.train()
    for epoch in range(1, EPOCHS+1):
        opt.zero_grad()
        emb   = model(data.x, data.edge_index)
        raw = head(emb[users], emb[hotels])
        preds = torch.sigmoid(raw)*5.0
        loss  = F.mse_loss(preds, target)
        loss.backward(); opt.step(); sched.step()

        if epoch % LOG_INTERVAL == 0 or epoch == EPOCHS:
            lr_now = opt.param_groups[0]["lr"]
            logging.info(f"Epoch {epoch:3d} | MSE {loss:.4f} | LR {lr_now:.0e}")

    # 7) save everything
    md = Path(cfg.model_dir); md.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(),    md / "graphsage.pth")
    torch.save(head.state_dict(),     md / "mlp_head.pth")
    joblib.dump(uenc,                 md / "user_encoder.pkl")
    joblib.dump(henc,                 md / "hotel_encoder.pkl")
    joblib.dump(hscaler,              md / "hotel_scaler.pkl")
    joblib.dump(uscaler,              md / "user_scaler.pkl")
    logging.info("Done – artifacts saved to %s", md)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True,
                   help="Path to enriched Train.csv")
    p.add_argument("--model_dir", default="models/",
                   help="Where to write models & scalers")
    cfg = p.parse_args()
    main(cfg)