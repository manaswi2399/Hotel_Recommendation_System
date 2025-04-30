import pandas as pd
import numpy as np
import argparse
from ast import literal_eval
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


def precision_at_k(recommended, relevant, k=5):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k=5):
    if len(relevant) == 0:
        return None
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def evaluate(test_path, predictions_path, k=5, rating_threshold=4):
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)
    test_df = test_df.rename(columns={"author": "user_id"})
    pred_df['recommended_hotels'] = pred_df['recommended_hotels'].apply(literal_eval)

    # Load user encoder
    user_encoder = joblib.load("trained_model_CF/user_encoder.pkl")

    precision_scores, recall_scores = [], []
    grouped = test_df[test_df['rating'] >= rating_threshold].groupby('user_id')['hotel_id'].apply(set).to_dict()

    y_true_all = []
    y_pred_all = []

    seen_users, new_users = set(), set()

    for row in pred_df.itertuples():
        user_id = row.user_id
        recommended = row.recommended_hotels
        relevant = grouped.get(user_id, set())

        if not relevant:
            continue

        if user_id in user_encoder:
            seen_users.add(user_id)
        else:
            new_users.add(user_id)

        # Collect scores
        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        if p is not None: precision_scores.append(p)
        if r is not None: recall_scores.append(r)

        # Build labels for confusion matrix
        for hotel_id in recommended[:k]:
            y_pred_all.append(1)  # predicted as relevant
            y_true_all.append(1 if hotel_id in relevant else 0)

        for hotel_id in relevant:
            if hotel_id not in recommended[:k]:
                y_pred_all.append(0)  # not recommended
                y_true_all.append(1)  # but actually relevant

    print(f"\nEvaluation Results (Top-{k}):")
    print(f"Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Recall@{k}:    {np.mean(recall_scores):.4f}")

    print("\nUser Stats:")
    print(f"Total users in test: {len(pred_df)}")
    print(f"Seen users (in training): {len(seen_users)}")
    print(f"New (cold-start) users: {len(new_users)}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Relevant', 'Not Relevant'])
    disp.plot(cmap='Blues', values_format='d')
    plt.savefig("confusion_matrix.png")
    return cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Top-K Precision, Recall and Confusion Matrix")
    parser.add_argument('--test', type=str, required=True, help='Path to test CSV with user ratings')
    parser.add_argument('--preds', type=str, required=True, help='Path to prediction CSV with top-k recommendations')
    parser.add_argument('--k', type=int, default=5, help='Top K value')
    parser.add_argument('--threshold', type=float, default=4.0, help='Rating threshold to define relevance')

    args = parser.parse_args()
    evaluate(args.test, args.preds, k=args.k, rating_threshold=args.threshold)
