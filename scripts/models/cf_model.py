import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import itertools

# Paths
train_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_with_topics_sentiment.csv'
val_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Validation_with_topics_sentiment.csv'

# Load data
train_df = pd.read_csv(train_path)
train_df = train_df.rename(columns={"author": "user_id"})

val_df = pd.read_csv(val_path)
val_df = val_df.rename(columns={"author": "user_id"})

train_df.columns = val_df.columns = train_df.columns.str.replace(' ', '_')

# Subrating columns
sub_rating_cols = ["cleanliness", "service", "rooms", "sleep_quality", "value", "location_rating"]

# Normalize
scalers = {}
scalers['rating'] = MinMaxScaler()
train_df['rating_norm'] = scalers['rating'].fit_transform(train_df[['rating']])

for col in sub_rating_cols:
    scalers[col] = MinMaxScaler()
    train_df[col+'_norm'] = scalers[col].fit_transform(train_df[[col]])

scalers['sentiment_score'] = MinMaxScaler()
train_df['sentiment_norm'] = scalers['sentiment_score'].fit_transform(train_df[['sentiment_score']])

rating_weight = 0.5
subrating_weight = 0.4 / len(sub_rating_cols)
sentiment_weight = 0.1

train_df['final_score'] = (
    rating_weight * train_df['rating_norm'] +
    subrating_weight * train_df[[col+'_norm' for col in sub_rating_cols]].sum(axis=1) +
    sentiment_weight * train_df['sentiment_norm']
)

# Encode users and hotels
user_encoder = {id_: idx for idx, id_ in enumerate(train_df['user_id'].unique())}
hotel_encoder = {id_: idx for idx, id_ in enumerate(train_df['hotel_id'].unique())}
train_df['user_idx'] = train_df['user_id'].map(user_encoder)
train_df['hotel_idx'] = train_df['hotel_id'].map(hotel_encoder)

# Train Sparse matrix
train_sparse = sparse.coo_matrix(
    (train_df['final_score'], (train_df['user_idx'], train_df['hotel_idx'])),
    shape=(len(user_encoder), len(hotel_encoder))
)

# Validation prep
val_df = val_df.rename(columns={"author": "user_id"})
val_df['rating_norm'] = scalers['rating'].transform(val_df[['rating']])
val_df['sentiment_norm'] = scalers['sentiment_score'].transform(val_df[['sentiment_score']])
for col in sub_rating_cols:
    val_df[col+'_norm'] = scalers[col].transform(val_df[[col]])

val_df['final_score'] = (
    rating_weight * val_df['rating_norm'] +
    subrating_weight * val_df[[col+'_norm' for col in sub_rating_cols]].sum(axis=1) +
    sentiment_weight * val_df['sentiment_norm']
)

val_df['user_idx'] = val_df['user_id'].map(user_encoder)
val_df['hotel_idx'] = val_df['hotel_id'].map(hotel_encoder)

# Hyperparameter grid
factors_grid = [64, 128]
regularization_grid = [0.1, 0.01]
iterations_grid = [20, 40]

param_grid = list(itertools.product(factors_grid, regularization_grid, iterations_grid))

# Results
results = []

for factors, reg, iters in param_grid:
    print(f"Training with factors={factors}, reg={reg}, iterations={iters}")
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, use_gpu=True)
    model.fit(train_sparse.T)

    als_preds = []
    cb_preds = []
    actual_ratings = []

    for row in val_df.itertuples():
        actual_ratings.append(row.rating)

        if pd.isna(row.user_idx) or pd.isna(row.hotel_idx):
            cb_score = (
                rating_weight * row.rating_norm +
                subrating_weight * sum(getattr(row, col + '_norm') for col in sub_rating_cols) +
                sentiment_weight * row.sentiment_norm
            )
            cb_preds.append(cb_score)
            als_preds.append(None)
        else:
            user_idx = int(row.user_idx)
            hotel_idx = int(row.hotel_idx)

            user_vector = sparse.csr_matrix(
                (np.array([1.0]), (np.array([0]), np.array([hotel_idx]))),
                shape=(1, len(hotel_encoder))
            )

            recommended = model.recommend(userid=0, user_items=user_vector, N=1, items=np.array([hotel_idx]), recalculate_user=True)

            if recommended and len(recommended) > 0:
              if isinstance(recommended[0], (tuple, list)) and len(recommended[0]) == 2:
                _, score = recommended[0]
              else:
                score = 0.0
            else:
                score = 0.0

            als_preds.append(score)
            cb_preds.append(None)

    # Normalize ALS predictions
    als_preds_array = np.array([p for p in als_preds if p is not None]).reshape(-1, 1)
    if als_preds_array.shape[0] == 0:
        continue

    als_preds_array = MinMaxScaler().fit_transform(als_preds_array)
    als_preds_array = scalers['rating'].inverse_transform(als_preds_array)
    als_actual = [r for p, r in zip(als_preds, actual_ratings) if p is not None]

    cb_preds_array = [p for p in cb_preds if p is not None]
    cb_actual = [r for p, r in zip(cb_preds, actual_ratings) if p is not None]

    combined_preds = [pred if pred is not None else cb for pred, cb in zip(als_preds, cb_preds)]

    als_rmse = np.sqrt(mean_squared_error(als_actual, als_preds_array))
    cb_rmse = np.sqrt(mean_squared_error(cb_actual, cb_preds_array))
    overall_rmse = np.sqrt(mean_squared_error(actual_ratings, combined_preds))

    print(f"ALS RMSE: {als_rmse:.4f}, CB RMSE: {cb_rmse:.4f}, Overall RMSE: {overall_rmse:.4f}")
    results.append((factors, reg, iters, als_rmse, cb_rmse, overall_rmse))


results = sorted(results, key=lambda x: x[5])
print("\nBest Hyperparameters:")
print(f"Factors: {results[0][0]}, Regularization: {results[0][1]}, Iterations: {results[0][2]}, Overall RMSE: {results[0][5]:.4f}")
