import argparse, logging
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn_train import GraphSAGE, build_graph, set_seeds

def load_gnn(model_dir, device):
    uenc  = joblib.load(Path(model_dir) / "user_encoder.pkl")
    henc  = joblib.load(Path(model_dir) / "hotel_encoder.pkl")
    sd    = torch.load(Path(model_dir) / "graphsage.pth", map_location="cpu")

    in_dim  = sd["conv1.lin_l.weight"].shape[1]
    hid_dim = sd["conv1.lin_l.weight"].shape[0]
    out_dim = sd["conv2.lin_l.weight"].shape[0]

    gnn = GraphSAGE(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim).to(device)
    gnn.load_state_dict(sd)
    gnn.eval()
    return gnn, uenc, henc

def vec_topk(mat, k):
    """Return lists of (row, col, val) for the top-k entries in each row."""
    topv, topi = mat.topk(k, dim=1)
    rows, cols, vals = [], [], []
    for r in range(mat.size(0)):
        rows.extend([r] * k)
        cols.extend(topi[r].tolist())
        vals.extend(topv[r].tolist())
    return rows, cols, vals

def main(cfg):
    logging.basicConfig(format="%(asctime)s %(levelname)s | %(message)s",
                        level=logging.INFO)
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GNN + encoders + SVD
    gnn, uenc, henc = load_gnn(cfg.model_dir, device)
    svd = joblib.load(Path(cfg.model_dir) / "svd.pkl")

    # Load the same CSV used for CF/GNN training
    interactions = pd.read_csv(cfg.train_csv).dropna(subset=["author", "hotel_id"])
    interactions["uid"] = uenc.transform(interactions["author"])
    interactions["hid"] = henc.transform(interactions["hotel_id"])

    # Build graph & node embeddings
    feat_cols = cfg.feat_cols.split(",")
    pyg_data, *_ = build_graph(interactions, feat_cols)
    emb = gnn(pyg_data.x.to(device), pyg_data.edge_index.to(device)).cpu()

    U = len(uenc.classes_)
    user_emb  = emb[:U]          # [U, D]
    hotel_emb = emb[U:]          # [H, D]
    H = hotel_emb.size(0)

    # Vectorized GNN top-K
    logging.info("GNN top-%d …", cfg.top_k)
    rows, cols, gnn_vals = [], [], []
    dl = DataLoader(torch.arange(U), batch_size=cfg.batch_size)
    for batch in dl:
        ue = user_emb[batch]          # [B, D]
        sc = ue @ hotel_emb.T         # [B, H]
        r, c, v = vec_topk(sc, cfg.top_k)
        rows += (batch[r]).tolist()
        cols += c
        gnn_vals += v
    gnn_df = pd.DataFrame({"uid": rows, "hid": cols, "gnn_score": gnn_vals})

    # Vectorized SVD top-K
    logging.info("SVD top-%d …", cfg.top_k)
    pu, qi = svd.pu, svd.qi
    trainset = svd.trainset
    u_inner = [trainset.to_inner_uid(u) for u in uenc.classes_]
    i_inner = [trainset.to_inner_iid(i) for i in henc.classes_]
    user_f = torch.tensor(pu[u_inner])     # [U, F]
    item_f = torch.tensor(qi[i_inner])     # [H, F]

    rows, cols, svd_vals = [], [], []
    for batch in dl:
        sc = user_f[batch] @ item_f.T
        r, c, v = vec_topk(sc, cfg.top_k)
        rows += (batch[r]).tolist()
        cols += c
        svd_vals += v
    svd_df = pd.DataFrame({"uid": rows, "hid": cols, "svd_score": svd_vals})

    # Combine GNN & SVD scores
    merged = pd.merge(gnn_df, svd_df, on=["uid", "hid"])
    merged["combined"] = 0.5 * merged.gnn_score + 0.5 * merged.svd_score

    # Keep top-K by ‘combined’ score
    topk = (
        merged.sort_values(["uid", "combined"], ascending=False)
              .groupby("uid")
              .head(cfg.top_k)
              .reset_index(drop=True)
    )

    ratings_df = interactions[["uid", "hid", "rating"]].dropna()

    # Identify rated pairs missing from `topk`
    missing = ratings_df.merge(topk[["uid", "hid"]], on=["uid", "hid"],
                               how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")

    if not missing.empty:
        logging.info("Adding %d rated (uid, hid) pairs missing from top-K list.",
                     len(missing))

        with torch.no_grad():                                     
            ue = user_emb[torch.tensor(missing.uid.values)]
            he = hotel_emb[torch.tensor(missing.hid.values)]
            missing["gnn_score"] = (ue * he).sum(dim=1).detach().cpu().numpy()   

            uf = user_f[torch.tensor(missing.uid.values)]
            hf = item_f[torch.tensor(missing.hid.values)]
            missing["svd_score"] = (uf * hf).sum(dim=1).detach().cpu().numpy() 

        missing["combined"] = 0.5 * missing.gnn_score + 0.5 * missing.svd_score
        topk = pd.concat([topk, missing], ignore_index=True)


    # De-duplicate in case of overlap
    topk = topk.drop_duplicates(subset=["uid", "hid"])

    # Attach hotel-level metadata
    hotel_meta = (
        interactions.groupby("hid")[feat_cols]
                    .mean()
                    .reset_index()
    )
    topk = topk.merge(hotel_meta, on="hid", how="left")

    # Attach ground-truth ratings 
    topk = topk.merge(ratings_df, on=["uid", "hid"], how="left")

    # Save
    Path(cfg.output_csv).parent.mkdir(exist_ok=True, parents=True)
    topk.to_csv(cfg.output_csv, index=False)
    logging.info("Wrote %d candidate rows → %s", len(topk), cfg.output_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--top_k",      type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_csv", default="models/candidates.csv")
    p.add_argument("--feat_cols",  default=
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