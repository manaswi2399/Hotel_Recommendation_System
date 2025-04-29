import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Paths
train_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_with_topics_sentiment.csv'
model_save_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/models/trained_model/cf_als_model.pkl'
user_encoder_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/models/trained_model/user_encoder.pkl'
hotel_encoder_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/models/trained_model/hotel_encoder.pkl'
scalers_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/models/trained_model/scalers.pkl'

# Load Train Data
df = pd.read_csv(train_path)

# Define subrating columns
sub_rating_cols = ["cleanliness", "service", "rooms", "sleep quality", "value", "location_rating"]

# Normalize
scalers = {}
scalers['rating'] = MinMaxScaler()
df['rating_norm'] = scalers['rating'].fit_transform(df[['rating']])

for col in sub_rating_cols:
    scalers[col] = MinMaxScaler()
    df[col + '_norm'] = scalers[col].fit_transform(df[[col]])

scalers['sentiment_score'] = MinMaxScaler()
df['sentiment_norm'] = scalers['sentiment_score'].fit_transform(df[['sentiment_score']])

# Blended final score
rating_weight = 0.5
subrating_weight = 0.4 / len(sub_rating_cols)
sentiment_weight = 0.1

df['final_score'] = (
    rating_weight * df['rating_norm'] +
    subrating_weight * df[[col + '_norm' for col in sub_rating_cols]].sum(axis=1) +
    sentiment_weight * df['sentiment_norm']
)

# Encode users and hotels
user_encoder = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
hotel_encoder = {id_: idx for idx, id_ in enumerate(df['hotel_id'].unique())}

user_decoder = {idx: id_ for id_, idx in user_encoder.items()}
hotel_decoder = {idx: id_ for idx, id_ in hotel_encoder.items()}

df['user_idx'] = df['user_id'].map(user_encoder)
df['hotel_idx'] = df['hotel_id'].map(hotel_encoder)

train_sparse = sparse.coo_matrix(
    (df['final_score'], (df['user_idx'], df['hotel_idx'])),
    shape=(len(user_encoder), len(hotel_encoder))
)
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20,
    use_gpu=True
)

print("Training ALS model...")
model.fit(train_sparse.T)

# Save model and encoders
print("Saving model and encoders...")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(model, model_save_path)
joblib.dump(user_encoder, user_encoder_path)
joblib.dump(hotel_encoder, hotel_encoder_path)
joblib.dump(scalers, scalers_path)

