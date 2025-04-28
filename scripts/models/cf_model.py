import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Multiply, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#DataFrame has columns: 'user_id', 'hotel_id', 'rating', 'review_text'
df = pd.read_csv('hotel_reviews.csv')

# Normalize ratings
scaler = MinMaxScaler()
df['rating_norm'] = scaler.fit_transform(df[['rating']])

# Encode user and hotel IDs
user_ids = df['user_id'].unique().tolist()
hotel_ids = df['hotel_id'].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
hotel2hotel_encoded = {x: i for i, x in enumerate(hotel_ids)}

df['user'] = df['user_id'].map(user2user_encoded)
df['hotel'] = df['hotel_id'].map(hotel2hotel_encoded)

num_users = len(user2user_encoded)
num_hotels = len(hotel2hotel_encoded)

#Train-val split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#Model parameters
embedding_size = 64
mlp_layers = [128, 64]
dropout_rate = 0.2
learning_rate = 0.001
batch_size = 256
epochs = 20

#Model architecture
user_input = Input(shape=(1,), name='user_input')
hotel_input = Input(shape=(1,), name='hotel_input')
sentiment_input = Input(shape=(1,), name='sentiment_input')

user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
hotel_embedding = Embedding(input_dim=num_hotels, output_dim=embedding_size, name='hotel_embedding')(hotel_input)

#GMF
gmf = Multiply()([user_embedding, hotel_embedding])
gmf = Flatten()(gmf)

#MLP
mlp = Concatenate()([Flatten()(user_embedding), Flatten()(hotel_embedding)])
for units in mlp_layers:
    mlp = Dense(units, activation='relu')(mlp)
    mlp = Dropout(dropout_rate)(mlp)

#Concatenate GMF and MLP parts with sentiment input
concat = Concatenate()([gmf, mlp, sentiment_input])

#Final prediction layer
output = Dense(1, activation='sigmoid', name='output')(concat)

model = Model(inputs=[user_input, hotel_input, sentiment_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

#Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#Training model
history = model.fit(
    [train_df['user'], train_df['hotel'], train_df['sentiment_score']],
    train_df['rating_norm'],
    validation_data=(
        [val_df['user'], val_df['hotel'], val_df['sentiment_score']],
        val_df['rating_norm']
    ),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

#Evaluating model
val_predictions = model.predict([val_df['user'], val_df['hotel'], val_df['sentiment_score']])
val_predictions = scaler.inverse_transform(val_predictions)
val_rmse = np.sqrt(mean_squared_error(val_df['rating'], val_predictions))
print(f'Validation RMSE: {val_rmse:.4f}')
