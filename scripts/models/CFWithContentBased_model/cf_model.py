
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import joblib
import os


train_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_with_topics_sentiment.csv'
val_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Validation_with_topics_sentiment.csv'
model_dir = 'trained_model_CF'
os.makedirs(model_dir, exist_ok=True)


train_df = pd.read_csv(train_path).sample(frac=0.016, random_state=42)
train_df= train_df.rename(columns={"author": "user_id"})
val_df = pd.read_csv(val_path).sample(frac=0.01, random_state=42)
val_df= val_df.rename(columns={"author": "user_id"})
train_df.columns = val_df.columns = train_df.columns.str.replace(' ', '_')


sub_rating_cols = ["cleanliness", "service", "rooms", "sleep_quality", "value", "location_rating"]
scalers = {'rating': MinMaxScaler(), 'sentiment_score': MinMaxScaler()}
train_df['rating_norm'] = scalers['rating'].fit_transform(train_df[['rating']])
train_df['sentiment_norm'] = scalers['sentiment_score'].fit_transform(train_df[['sentiment_score']])
for col in sub_rating_cols:
    scalers[col] = MinMaxScaler()
    train_df[col + '_norm'] = scalers[col].fit_transform(train_df[[col]])


rating_weight = 0.6
subrating_weight = 0.4 / len(sub_rating_cols)
sentiment_weight = 0.2


subrating_sum = np.zeros(len(train_df))
for col in sub_rating_cols:
    is_missing_col = f"is_{col}_missing"
    weighted_col = np.where(train_df[is_missing_col] == 1, 0.1 * train_df[col + '_norm'], train_df[col + '_norm'])
    subrating_sum += weighted_col

train_df['final_score'] = (
    rating_weight * train_df['rating_norm'] +
    subrating_weight * subrating_sum +
    sentiment_weight * train_df['sentiment_norm']
)


user_encoder = {id_: idx for idx, id_ in enumerate(train_df['user_id'].unique())}
hotel_encoder = {id_: idx for idx, id_ in enumerate(train_df['hotel_id'].unique())}
train_df['user_idx'] = train_df['user_id'].map(user_encoder)
train_df['hotel_idx'] = train_df['hotel_id'].map(hotel_encoder)


train_sparse = sparse.coo_matrix(
    (train_df['final_score'], (train_df['user_idx'], train_df['hotel_idx'])),
    shape=(len(user_encoder), len(hotel_encoder))
)


val_df['rating_norm'] = scalers['rating'].transform(val_df[['rating']])
val_df['sentiment_norm'] = scalers['sentiment_score'].transform(val_df[['sentiment_score']])
for col in sub_rating_cols:
    val_df[col + '_norm'] = scalers[col].transform(val_df[[col]])

subrating_sum_val = np.zeros(len(val_df))
for col in sub_rating_cols:
    is_missing_col = f"is_{col}_missing"
    weighted_col = np.where(val_df[is_missing_col] == 1, 0.1 * val_df[col + '_norm'], val_df[col + '_norm'])
    subrating_sum_val += weighted_col

val_df['final_score'] = (
    rating_weight * val_df['rating_norm'] +
    subrating_weight * subrating_sum_val +
    sentiment_weight * val_df['sentiment_norm']
)

val_df['user_idx'] = val_df['user_id'].map(user_encoder)
val_df['hotel_idx'] = val_df['hotel_id'].map(hotel_encoder)


cb_features = [
    "cleanliness", "service", "rooms", "sleep_quality", "value", "location_rating", "sentiment_score",
    "has_room_quality", "has_location", "has_staff_service", "has_cleanliness", "has_amenities",
    "has_value_for_money", "has_food_and_restaurant", "has_noise_level", "has_comfort"
]
X_cb = train_df[cb_features]
y_cb = train_df['rating']
cb_model = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), Ridge(alpha=1.0))
cb_model.fit(X_cb, y_cb)


factors = 64
reg = 0.01
iters = 20


print(f"Training ALS model with factors={factors}, reg={reg}, iterations={iters}")
model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, use_gpu=True)
model.fit(train_sparse.T)
als_preds, cb_preds, actual_ratings = [], [], []

for row in val_df.itertuples():
    actual_ratings.append(row.rating)
    if pd.isna(row.user_idx) or pd.isna(row.hotel_idx):
        features = []
        for col in cb_features:
            missing_flag = f"is_{col}_missing"
            val = getattr(row, col)
            is_missing = getattr(row, missing_flag, 0)
            features.append(0.1 * val if is_missing == 1 else val)
        cb_score = cb_model.predict([features])[0]
        cb_preds.append(cb_score)
        als_preds.append(None)
    else:
        user_vector = sparse.csr_matrix((np.array([1.0]), ([0], [int(row.hotel_idx)])), shape=(1, len(hotel_encoder)))
        recommended = model.recommend(userid=0, user_items=user_vector, N=1, items=np.array([int(row.hotel_idx)]), recalculate_user=True)
        if recommended and len(recommended) > 0:
            if isinstance(recommended[0], (tuple, list)) and len(recommended[0]) == 2:
                _, score = recommended[0]
            else:
                score = 0.0
        else:
            score = 0.0
        als_preds.append(score)
        cb_preds.append(None)

als_preds_array = np.array([p for p in als_preds if p is not None]).reshape(-1, 1)
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


joblib.dump(model, os.path.join(model_dir, 'cf_als_model.pkl'))
joblib.dump(cb_model, os.path.join(model_dir, 'cb_fallback_model.pkl'))
joblib.dump(scalers, os.path.join(model_dir, 'scalers.pkl'))
joblib.dump(user_encoder, os.path.join(model_dir, 'user_encoder.pkl'))
joblib.dump(hotel_encoder, os.path.join(model_dir, 'hotel_encoder.pkl'))
print("\nAll models and encoders saved successfully.")


val_users = set(val_df['user_id'])
train_users = set(user_encoder.keys())
new_users = val_users - train_users
print(f"Total validation users: {len(val_users)}")
print(f"Users seen in training: {len(val_users) - len(new_users)}")
print(f"New (cold-start) users: {len(new_users)}")




test_path = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Test_with_topics_sentiment.csv'
model_dir = 'trained_model_CF'
output_path = os.path.join(model_dir, 'test_top5_predictions.csv')
top_k = 5
sample_size = 5000

test_df_full = pd.read_csv(test_path).sample(frac=0.002, random_state=42)


test_df_full = test_df_full.rename(columns={"author": "user_id"})
test_df_full.columns = test_df_full.columns.str.replace(' ', '_')

sample_size = min(sample_size, len(test_df_full))

test_df = test_df_full.sample(n=sample_size, random_state=42).reset_index(drop=True)


sub_rating_cols = ["cleanliness", "service", "rooms", "sleep_quality", "value", "location_rating"]
test_df['rating_norm'] = scalers['rating'].transform(test_df[['rating']])
test_df['sentiment_norm'] = scalers['sentiment_score'].transform(test_df[['sentiment_score']])
for col in sub_rating_cols:
    test_df[col + '_norm'] = scalers[col].transform(test_df[[col]])


rating_weight = 0.6
subrating_weight = 0.4 / len(sub_rating_cols)
sentiment_weight = 0.2
subrating_sum_test = np.zeros(len(test_df))
for col in sub_rating_cols:
    is_missing_col = f"is_{col}_missing"
    weighted_col = np.where(test_df[is_missing_col] == 1, 0.1 * test_df[col + '_norm'], test_df[col + '_norm'])
    subrating_sum_test += weighted_col

test_df['final_score'] = (
    rating_weight * test_df['rating_norm'] +
    subrating_weight * subrating_sum_test +
    sentiment_weight * test_df['sentiment_norm']
)


test_df['user_idx'] = test_df['user_id'].map(user_encoder)
test_df['hotel_idx'] = test_df['hotel_id'].map(hotel_encoder)


user_seen_hotels = train_df.groupby('user_id')['hotel_id'].apply(set).to_dict()



hotel_decoder = {v: k for k, v in hotel_encoder.items()}

user_top5_recs = {}

for user_id in test_df['user_id'].unique():
    seen_hotels = user_seen_hotels.get(user_id, set())

    if user_id in user_encoder:
        uidx = user_encoder[user_id]
        seen_encoded = {int(hotel_encoder[h]) for h in seen_hotels if h in hotel_encoder}
        user_items_matrix = train_sparse.tocsr()[uidx]

        recommended_ids, _ = model.recommend(
            userid=uidx,
            user_items=user_items_matrix,
            N=top_k + len(seen_encoded),
            filter_items=list(seen_encoded),
            recalculate_user=True
        )

        hotel_ids = [
            hotel_decoder[int(hid)]
            for hid in recommended_ids
            if int(hid) in hotel_decoder and int(hid) not in seen_encoded
        ][:top_k]

        user_top5_recs[user_id] = hotel_ids

    else:
        candidate_hotels = list(set(test_df['hotel_id'].unique()) - seen_hotels)
        hotel_scores = []
        for hotel_id in candidate_hotels:
            row = test_df[(test_df['user_id'] == user_id) & (test_df['hotel_id'] == hotel_id)]
            if row.empty:
                continue
            row = row.iloc[0]
            features = [0.1 * row[col] if row.get(f'is_{col}_missing', 0) == 1 else row[col] for col in cb_features]
            score = cb_model.predict([features])[0]
            hotel_scores.append((hotel_id, score))
        top_hotels = sorted(hotel_scores, key=lambda x: x[1], reverse=True)[:top_k]
        user_top5_recs[user_id] = [hid for hid, _ in top_hotels]




pred_df = pd.DataFrame([
    {'user_id': user, 'recommended_hotels': recs}
    for user, recs in user_top5_recs.items()
])
pred_df.to_csv(output_path, index=False)
print(f"\nTop-5 recommendations saved to {output_path}")

test_users = set(test_df['user_id'])
test_user = set(user_encoder.keys())
new_user = test_users - test_user
print(f"Total test users: {len(test_users)}")
print(f"Users seen in training: {len(test_users) - len(new_user)}")
print(f"New (cold-start) users: {len(new_user)}")
