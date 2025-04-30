"""
Print head and distribution of rows-per-user in topN_final.csv
"""
import argparse, pandas as pd

def main(args):
    df = pd.read_csv(args.topn_csv)
    print("\nFIRST 5 ROWS\n------------")
    print(df.head())
    print("\nROWS / USER\n-----------")
    print(df.groupby("uid").size().value_counts().head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn_csv", required=True)
    main(ap.parse_args())