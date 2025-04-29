import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Load train and validation datasets
train_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_with_topics_sentiment.csv'
val_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Validation_with_topics_sentiment.csv'

df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

#Define subrating columns
sub_rating_cols = ["cleanliness", "service", "rooms", "sleep quality", "value", "location_rating"]

#Normalize main rating and all subratings
scalers = {}

#Normalize main rating
scalers['rating'] = MinMaxScaler()
df['rating_norm'] = scalers['rating'].fit_transform(df[['rating']])

#Normalize subratings
for col in sub_rating_cols:
    scalers[col] = MinMaxScaler()
    df[col + '_norm'] = scalers[col].fit_transform(df[[col]])

#Normalize sentiment_score
scalers['sentiment_score'] = MinMaxScaler()
df['sentiment_norm'] = scalers['sentiment_score'].fit_transform(df[['sentiment_score']])

#Blending
rating_weight = 0.5
subrating_weight = 0.4 / len(sub_rating_cols)
sentiment_weight = 0.1

df['final_score'] = (
    rating_weight * df['rating_norm'] +
    subrating_weight * df[[col + '_norm' for col in sub_rating_cols]].sum(axis=1) +
    sentiment_weight * df['sentiment_norm']
)

#Encode users and hotels
user_encoder = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
hotel_encoder = {id_: idx for idx, id_ in enumerate(df['hotel_id'].unique())}

user_decoder = {idx: id_ for id_, idx in user_encoder.items()}
hotel_decoder = {idx: id_ for id_, idx in hotel_encoder.items()}

df['user_idx'] = df['user_id'].map(user_encoder)
df['hotel_idx'] = df['hotel_id'].map(hotel_encoder)

#Preprocess validation set the same way
val_df['rating_norm'] = scalers['rating'].transform(val_df[['rating']])

for col in sub_rating_cols:
    val_df[col + '_norm'] = scalers[col].transform(val_df[[col]])

val_df['sentiment_norm'] = scalers['sentiment_score'].transform(val_df[['sentiment_score']])

val_df['final_score'] = (
    rating_weight * val_df['rating_norm'] +
    subrating_weight * val_df[[col + '_norm' for col in sub_rating_cols]].sum(axis=1) +
    sentiment_weight * val_df['sentiment_norm']
)

val_df['user_idx'] = val_df['user_id'].map(user_encoder)
val_df['hotel_idx'] = val_df['hotel_id'].map(hotel_encoder)

#Create sparse matrices
train_sparse = sparse.coo_matrix(
    (df['final_score'], (df['user_idx'], df['hotel_idx'])),
    shape=(len(user_encoder), len(hotel_encoder))
)

#ALS model
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20,
    use_gpu=True
)

print("Training ALS model...")
model.fit(train_sparse.T)

#Validation predictions
print("Predicting on validation set...")
val_preds = []
actual_ratings = []

for row in val_df.itertuples():
    u = row.user_idx
    h = row.hotel_idx
    actual = row.rating

    if pd.isna(u) or pd.isna(h):
        continue

    u = int(u)
    h = int(h)

    pred_norm = model.rank_items(
        u,
        train_sparse.T,
        np.array([h])
    )[0][1]

    val_preds.append(pred_norm)
    actual_ratings.append(actual)

#Normalize predictions
val_preds = np.array(val_preds).reshape(-1, 1)
val_preds = MinMaxScaler().fit_transform(val_preds)
val_preds = scalers['rating'].inverse_transform(val_preds)

#RMSE
val_rmse = np.sqrt(mean_squared_error(actual_ratings, val_preds))
print(f'Validation RMSE: {val_rmse:.4f}')