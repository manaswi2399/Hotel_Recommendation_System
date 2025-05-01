import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/raw/HotelDataset_CleanedBeforeSplit.csv")

# Drop year 2019 (incomplete data)
df = df[df["year"] < 2019]

# Splits
train_df = df[df["year"].between(2004, 2016)]
val_df   = df[df["year"] == 2017]
test_df  = df[df["year"] == 2018]

# Sub-rating columns to impute
sub_rating_cols = ["cleanliness", "service", "rooms", "sleep quality", "value", "location_rating"]

# Compute hotel-wise means from training set
hotel_means = train_df.groupby("hotel_url")[sub_rating_cols].mean()

def fill_with_train_means(df_part):
    df_part = df_part.copy()
    for col in sub_rating_cols:
        df_part[col] = df_part[col].fillna(df_part["hotel_url"].map(hotel_means[col]))
    return df_part


train_df = fill_with_train_means(train_df)
val_df   = fill_with_train_means(val_df)
test_df  = fill_with_train_means(test_df)


train_df.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/processed/Train.csv", index=False)
val_df.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/processed/Validation.csv", index=False)
test_df.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/processed/Test.csv", index=False)

print("Data split saved successfully!")
