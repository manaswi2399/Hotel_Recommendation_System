import argparse, logging, math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)

#Model
class AttnReranker(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)

    def forward(self, x):                 # x: (N, dim)
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)         # (N,)

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std() + 1e-6)

def build_feat_tensor(df: pd.DataFrame, cols, device):
    return torch.tensor(zscore(df[cols]).values, dtype=torch.float, device=device)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df     = pd.read_csv(cfg.candidates_csv)
    
    if "rating" not in df.columns:
        rating_cols = [c for c in df.columns if c.startswith("rating")]
        if rating_cols:
            df["rating"] = df[rating_cols[0]]

    feat_cols = [
        "combined",
        "cleanliness","service","rooms","value","location_rating",
        "sentiment_score",
        "has_room_quality","has_location","has_staff_service",
        "has_cleanliness","has_amenities","has_value_for_money",
        "has_food_and_restaurant","has_noise_level","has_comfort",
        "is_cleanliness_missing","is_service_missing",
        "is_rooms_missing","is_sleep_quality_missing",
        "is_value_missing","is_location_rating_missing"
    ]

    # full set for inference
    feats_full = build_feat_tensor(df, feat_cols, device)

    # training subset 
    train_df = df.dropna(subset=["rating"]).copy()
    if cfg.mode == "train" and train_df.empty:
        logging.error("No rows with ground-truth ratings – cannot train.")
        return

    feats_train = build_feat_tensor(train_df, feat_cols, device) if not train_df.empty else None
    y_train     = torch.tensor(train_df["rating"].values, dtype=torch.float, device=device) if not train_df.empty else None

    model = AttnReranker(dim=len(feat_cols)).to(device)

    # Train
    if cfg.mode == "train":
        logging.info("Training reranker … (%d rows with ratings)", len(train_df))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for ep in range(1, cfg.epochs + 1):
            model.train()
            opt.zero_grad()
            pred = model(feats_train)
            loss = F.mse_loss(pred, y_train)
            if torch.isnan(loss):
                logging.error("NaN loss at epoch %d – abort.", ep)
                return
            loss.backward(); opt.step()
            if ep % cfg.log_interval == 0 or ep == cfg.epochs:
                logging.info("Epoch %02d – train MSE %.4f", ep, loss.item())
        Path(cfg.model_dir).mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), Path(cfg.model_dir)/"attn.pth")

    else:  # infer mode
        model.load_state_dict(torch.load(Path(cfg.model_dir)/"attn.pth", map_location=device))
        model.eval()

    #Inference over ALL rows
    model.eval()
    with torch.no_grad():
        df["rerank_score"] = model(feats_full).cpu().numpy()

    #Top-N per user
    topN = (
        df.sort_values(["uid","rerank_score"], ascending=False)
          .groupby("uid")
          .head(cfg.top_n)
          .reset_index(drop=True)
    )
    Path(cfg.output_csv).parent.mkdir(exist_ok=True, parents=True)
    topN.to_csv(cfg.output_csv, index=False)
    logging.info("Top-%d saved to %s", cfg.top_n, cfg.output_csv)

    #Fit metrics on training subset 
    if not train_df.empty:
        with torch.no_grad():
            preds_train = model(feats_train).cpu().numpy()
        rmse = math.sqrt(mean_squared_error(train_df["rating"], preds_train))
        mae  = mean_absolute_error(train_df["rating"], preds_train)
        logging.info("Fitted RMSE %.4f | MAE %.4f on %d pairs", rmse, mae, len(train_df))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates_csv", default="models/candidates.csv")
    p.add_argument("--model_dir",      default="models/")
    p.add_argument("--mode",           choices=["train","infer"], default="train")
    p.add_argument("--epochs",         type=int,   default=15)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--log_interval",   type=int,   default=5)
    p.add_argument("--top_n",          type=int,   default=10)
    p.add_argument("--output_csv",     default="models/topN_final.csv")
    cfg = p.parse_args()
    main(cfg)