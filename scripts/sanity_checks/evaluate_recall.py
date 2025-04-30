import argparse, numpy as np, pandas as pd, joblib
from collections import defaultdict
from math import log2


def dcg(gains):
    return sum(g / log2(i + 2) for i, g in enumerate(gains))


def map_with_nan(series, mapping):
    return series.map(mapping).astype("float")


def main(cfg):
    topn = pd.read_csv(cfg.topn_csv)[["uid", "hid"]]

    uenc = joblib.load(cfg.user_enc)
    henc = joblib.load(cfg.hotel_enc)
    user2id  = {u: i for i, u in enumerate(uenc.classes_)}
    hotel2id = {h: i for i, h in enumerate(henc.classes_)}

    test = (pd.read_csv(cfg.test_csv, usecols=["author", "hotel_id", "rating"])
              .assign(uid=lambda d: map_with_nan(d.author,  user2id),
                      hid=lambda d: map_with_nan(d.hotel_id, hotel2id))
              .dropna(subset=["uid", "hid"]))         

    rel_test = test[test.rating >= cfg.threshold]

    rel_by_user = defaultdict(set)
    for r in rel_test.itertuples():
        rel_by_user[int(r.uid)].add(int(r.hid))

    K = cfg.k
    hits, total_rel, ndcg_sum, users = 0, 0, 0.0, 0
    for uid, group in topn.groupby("uid"):
        recs = group.hid.tolist()[:K]
        rels = rel_by_user.get(uid, set())
        if not rels:
            continue
        users += 1
        hit_vec = [1 if h in rels else 0 for h in recs]
        hits += sum(hit_vec)
        total_rel += len(rels)
        ideal = dcg([1] * min(len(rels), K))
        ndcg_sum += dcg(hit_vec) / ideal if ideal else 0

    recall = hits / total_rel if total_rel else 0
    ndcg   = ndcg_sum / users if users else 0

    print(f"Users in test file        : {test.author.nunique()}")
    print(f"Users evaluated (have recs): {users}")
    print(f"Recall@{K}                : {recall:.4f}")
    print(f"NDCG@{K}                  : {ndcg:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn_csv",   required=True)
    ap.add_argument("--test_csv",   required=True)
    ap.add_argument("--user_enc",   required=True)
    ap.add_argument("--hotel_enc",  required=True)
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=3.5)
    main(ap.parse_args())