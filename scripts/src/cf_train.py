"""
Train a Surprise SVD model on Train.csv and save to models/svd.pkl
"""
import argparse, logging, joblib
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)

def main(cfg):
    df = pd.read_csv(cfg.train_csv)[["author", "hotel_id", "rating"]]
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()  

    algo = SVD(n_factors=100, random_state=42, lr_all=0.005, reg_all=0.02)
    logging.info("Training SVD...")
    algo.fit(trainset)

    Path(cfg.model_dir).mkdir(exist_ok=True, parents=True)
    joblib.dump(algo, Path(cfg.model_dir) / "svd.pkl")
    logging.info("Saved SVD to %s", cfg.model_dir)

if __name__ == "__main__":
    import sys
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="processed Train.csv")
    p.add_argument("--model_dir", default="models/")
    cfg = p.parse_args(sys.argv[1:])
    main(cfg)