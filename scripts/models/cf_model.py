import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_with_topics_sentiment.csv')
val_df = pd.read_csv('/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Validation_with_topics_sentiment.csv')

#Normalize training ratings
rating_scaler = MinMaxScaler()
df['rating_norm'] = rating_scaler.fit_transform(df[['rating']])

#Normalize sentiment
if df['sentiment_score'].max() > 1.0:
    sentiment_scaler = MinMaxScaler()
    df['sentiment_norm'] = sentiment_scaler.fit_transform(df[['sentiment_score']])
else:
    df['sentiment_norm'] = df['sentiment_score']

#Blend rating and sentiment
alpha = 0.7
df['final_score'] = alpha * df['rating_norm'] + (1 - alpha) * df['sentiment_norm']

#Encode user and hotel IDs
user_encoder = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
hotel_encoder = {id_: idx for idx, id_ in enumerate(df['hotel_id'].unique())}

user_decoder = {idx: id_ for id_, idx in user_encoder.items()}
hotel_decoder = {idx: id_ for id_, idx in hotel_encoder.items()}

df['user_idx'] = df['user_id'].map(user_encoder)
df['hotel_idx'] = df['hotel_id'].map(hotel_encoder)

#Preprocess validation set
val_df['rating_norm'] = rating_scaler.transform(val_df[['rating']])

if val_df['sentiment_score'].max() > 1.0:
    val_df['sentiment_norm'] = sentiment_scaler.transform(val_df[['sentiment_score']])
else:
    val_df['sentiment_norm'] = val_df['sentiment_score']

val_df['final_score'] = alpha * val_df['rating_norm'] + (1 - alpha) * val_df['sentiment_norm']
val_df['user_idx'] = val_df['user_id'].map(user_encoder)
val_df['hotel_idx'] = val_df['hotel_id'].map(hotel_encoder)

#Create sparse matrices
train_sparse = sparse.coo_matrix(
    (df['final_score'], (df['user_idx'], df['hotel_idx'])),
    shape=(len(user_encoder), len(hotel_encoder))
)

val_sparse = sparse.coo_matrix(
    (val_df['final_score'], (val_df['user_idx'], val_df['hotel_idx'])),
    shape=(len(user_encoder), len(hotel_encoder))
)

#ALS model
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20,
    use_gpu=False
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

    if np.isnan(u) or np.isnan(h):
        continue  # skip unknown users/hotels

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
val_preds = rating_scaler.inverse_transform(val_preds)

#RMSE
val_rmse = np.sqrt(mean_squared_error(actual_ratings, val_preds))
print(f'Validation RMSE: {val_rmse:.4f}')
