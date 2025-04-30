import argparse, pandas as pd, joblib, numpy as np
from collections import defaultdict

def load_ids(topn, test, uenc, henc):
    topn = topn[["uid","hid"]]
    map_u = {u:i for i,u in enumerate(uenc.classes_)}
    map_h = {h:i for i,h in enumerate(henc.classes_)}
    test["uid"] = test.author.map(map_u);  test["hid"] = test.hotel_id.map(map_h)
    return topn, test.dropna(subset=["uid","hid"])

def run(topn, test, k):
    relevant = test[test.rating>=3.5].groupby("uid")["hid"].apply(set)
    users = 0; hits=0; ap_sum=0; covered_items=set()
    for uid, recs in topn.groupby("uid"):
        recs = recs.hid.tolist()[:k]
        rels = relevant.get(uid,set())
        if not rels: continue
        users += 1
        covered_items.update(recs)
        hit_vec=[1 if h in rels else 0 for h in recs]
        hits += any(hit_vec)
        # MAP
        num_hits=0; ap=0
        for i,hit in enumerate(hit_vec,1):
            if hit:
                num_hits+=1
                ap+=num_hits/i
        ap_sum += ap/max(len(rels),1)
    return dict(
        Users_eval=users,
        HitRate = hits/users if users else 0,
        MAP     = ap_sum/users if users else 0,
        Coverage= len(covered_items)/topn.hid.nunique()
    )

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--topn_csv",required=True)
    ap.add_argument("--test_csv",required=True)
    ap.add_argument("--user_enc",required=True)
    ap.add_argument("--hotel_enc",required=True)
    ap.add_argument("-k",type=int,default=20)
    args=ap.parse_args()

    topn=pd.read_csv(args.topn_csv)
    test=pd.read_csv(args.test_csv, usecols=["author","hotel_id","rating"])
    uenc=joblib.load(args.user_enc); henc=joblib.load(args.hotel_enc)
    topn,test=load_ids(topn,test,uenc,henc)
    metrics=run(topn,test,args.k)
    for k,v in metrics.items(): print(f"{k:10}: {v:.4f}" if isinstance(v,float) else f"{k:10}: {v}")