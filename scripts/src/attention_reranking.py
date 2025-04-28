import argparse, logging, math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)

class AttnReranker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)

    def forward(self, x):                    
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)             

def build_features(df, feat_cols):
    # normalise numerical columns
    x = df[feat_cols].copy()
    x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.tensor(x.values, dtype=torch.float)

def train(model, feats, y, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        opt.zero_grad()
        pred = model(feats)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        if ep % 2 == 0:
            logging.info("  ep%02d  train MSE %.4f", ep, loss.item())

def main(cfg):
    df = pd.read_csv(cfg.candidates_csv)

    feat_cols = ["combined","rating","cleanliness","service","rooms",
                 "value","location_rating","sentiment_score",
                 "topic_id","review_length"]
    feats = build_features(df, feat_cols)
    ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    model = AttnReranker(dim=feats.shape[1])
    if cfg.mode == "train":
        logging.info("Training attention reranker …")
        train(model, feats, ratings, epochs=cfg.epochs)
        torch.save(model.state_dict(), Path(cfg.model_dir) / "attn.pth")
    else:
        model.load_state_dict(torch.load(Path(cfg.model_dir) / "attn.pth"))
        model.eval()

    with torch.no_grad():
        scores = model(feats).numpy()

    df["rerank_score"] = scores
    topN = (
        df.sort_values(["uid","rerank_score"], ascending=False)
          .groupby("uid")
          .head(cfg.top_n)
    )
    out = Path(cfg.output_csv)
    topN.to_csv(out, index=False)
    logging.info("Top-%d per user saved to %s", cfg.top_n, out)

    # metrics vs. rating
    rmse = math.sqrt(mean_squared_error(df["rating"], scores))
    mae  = mean_absolute_error(df["rating"], scores)
    logging.info("Overall RMSE %.4f | MAE %.4f", rmse, mae)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates_csv", default="models/candidates.csv")
    p.add_argument("--model_dir", default="models/")
    p.add_argument("--mode", choices=["train","infer"], default="train")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--output_csv", default="models/topN_final.csv")
    cfg = p.parse_args()
    main(cfg)
