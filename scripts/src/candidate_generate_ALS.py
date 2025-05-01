import argparse, logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from gnn_train import GraphSAGE, build_graph, set_seeds

import implicit
import types
if not hasattr(implicit, 'gpu') or not hasattr(implicit.gpu, 'Matrix'):
    implicit.gpu = types.SimpleNamespace(Matrix=lambda arr: arr)


def load_gnn(model_dir, device):
    uenc = joblib.load(Path(model_dir) / "user_encoder.pkl")
    henc = joblib.load(Path(model_dir) / "hotel_encoder.pkl")
    sd = torch.load(Path(model_dir) / "graphsage.pth", map_location="cpu")

    in_dim = sd["conv1.lin_l.weight"].shape[1]
    model = GraphSAGE(in_dim=in_dim, hid_dim=sd["conv1.lin_l.weight"].shape[0], out_dim=sd["conv2.lin_l.weight"].shape[0])
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, uenc, henc


def vec_topk(mat, k):
    topv, topi = mat.topk(k, dim=1)
    rows, cols, vals = [], [], []
    for r in range(mat.size(0)):
        rows.extend([r] * k)
        cols.extend(topi[r].tolist())
        vals.extend(topv[r].tolist())
    return rows, cols, vals


def main(cfg):
    logging.basicConfig(format="%(asctime)s %(levelname)s | %(message)s", level=logging.INFO)
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    gnn, uenc, henc = load_gnn(cfg.model_dir, device)
    cf_model = joblib.load(Path(cfg.model_dir) / cfg.cf_model)
    cb_model = joblib.load(Path(cfg.model_dir) / cfg.cb_model)

    # Load interactions
    interactions = pd.read_csv(cfg.train_csv).dropna(subset=["author", "hotel_id"])
    interactions["uid"] = uenc.transform(interactions["author"])
    interactions["hid"] = henc.transform(interactions["hotel_id"])

    # Build graph & compute embeddings
    feat_cols = cfg.feat_cols.split(',')
    pyg_data, _, _, _, _ = build_graph(interactions, feat_cols)
    emb = gnn(pyg_data.x.to(device), pyg_data.edge_index.to(device)).cpu()

    # Split embeddings
    U = len(uenc.classes_)
    user_emb = emb[:U]
    hotel_emb = emb[U:]
    H = hotel_emb.size(0)

    # 1) GNN top-K
    logging.info("GNN top-%d …", cfg.top_k)
    rows_g, cols_g, g_vals = [], [], []
    for batch in DataLoader(torch.arange(U), batch_size=cfg.batch_size):
        ue = user_emb[batch]
        sc = ue @ hotel_emb.T
        r, c, v = vec_topk(sc, cfg.top_k)
        rows_g += (batch[r]).tolist()
        cols_g += c
        g_vals += v
    gnn_df = pd.DataFrame({"uid": rows_g, "hid": cols_g, "gnn_score": g_vals})

    # 2) CF-ALS top-K
    logging.info("CF-ALS top-%d …", cfg.top_k)
    cf_rows, cf_cols, cf_vals = [], [], []
    uf = cf_model.user_factors
    hf = cf_model.item_factors
    for uid in range(uf.shape[0]):
        scores = uf[uid] @ hf.T
        top_idx = np.argpartition(scores, -cfg.top_k)[-cfg.top_k:]
        cf_rows += [uid] * cfg.top_k
        cf_cols += top_idx.tolist()
        cf_vals += scores[top_idx].tolist()
    cf_df = pd.DataFrame({"uid": cf_rows, "hid": cf_cols, "cf_score": cf_vals})

    # 3) CB-fallback top-K
    logging.info("CB fallback top-%d …", cfg.top_k)
    cb_rows, cb_cols, cb_vals = [], [], []
    try:
        ubr = cb_model.user_factors
        hbr = cb_model.item_factors
        for uid in range(ubr.shape[0]):
            scores = ubr[uid] @ hbr.T
            top_idx = np.argpartition(scores, -cfg.top_k)[-cfg.top_k:]
            cb_rows += [uid] * cfg.top_k
            cb_cols += top_idx.tolist()
            cb_vals += scores[top_idx].tolist()
    except AttributeError:
        for uid in range(U):
            scores = [cb_model.predict(uid, hid) for hid in range(H)]
            top_idx = np.argpartition(scores, -cfg.top_k)[-cfg.top_k:]
            cb_rows += [uid] * cfg.top_k
            cb_cols += top_idx.tolist()
            cb_vals += [scores[i] for i in top_idx]
    cb_df = pd.DataFrame({"uid": cb_rows, "hid": cb_cols, "cb_score": cb_vals})

    # 4) Combine and select top-K by combined score
    merged = gnn_df.merge(cf_df, on=["uid","hid"]).merge(cb_df, on=["uid","hid"])
    merged["combined"] = (merged.gnn_score + merged.cf_score + merged.cb_score) / 3
    topk = merged.sort_values(["uid","combined"], ascending=False).groupby("uid").head(cfg.top_k)

    # Attach metadata and ratings
    hotel_meta = interactions.groupby("hid")[feat_cols].mean().reset_index()
    ratings_df = interactions[["uid","hid","rating"]].dropna()
    topk = topk.merge(hotel_meta, on="hid", how="left").merge(ratings_df, on=["uid","hid"], how="left")

    # Save
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
    topk.to_csv(cfg.output_csv, index=False)
    logging.info("Saved candidates to %s", cfg.output_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--cf_model", default="cf_als_model.pkl")
    p.add_argument("--cb_model", default="cb_fallback_model.pkl")
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_csv", default="models/candidates.csv")
    p.add_argument("--feat_cols", default=
        "cleanliness,service,rooms,value,location_rating,"  
        "sentiment_score,has_room_quality,has_location,"
        "has_staff_service,has_cleanliness,has_amenities,"
        "has_value_for_money,has_food_and_restaurant,"
        "has_noise_level,has_comfort,"
        "is_cleanliness_missing,is_service_missing,"
        "is_rooms_missing,is_sleep_quality_missing,"
        "is_value_missing,is_location_rating_missing")
    cfg = p.parse_args()
    main(cfg)