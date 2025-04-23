import pandas as pd

# Load the dataset
df = pd.read_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/raw/Hotel_Dataset_Without_duplicates.csv")

# Drop rows with missing author, rating, text, hotel_url
df.dropna(subset=["author", "rating", "text", "hotel_url"], inplace=True)

df["author"] = df["author"].str.strip()
df["title"] = df["title"].str.strip()
df["text"] = df["text"].str.strip()

df["title"] = df["title"].fillna("")

df["title"] = df["title"].str.lower()
df["text"] = df["text"].str.lower()


if "location" in df.columns:
    df.rename(columns={"location": "location_rating"}, inplace=True)
    
    

sub_rating_cols = ["cleanliness", "service", "rooms", "sleep quality", "value", "location_rating"]

# Convert sub-ratings to numeric and fill missing values with hotel-wise mean
for col in sub_rating_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df.groupby("hotel_url")[col].transform("mean"))

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")


df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["season"] = df["month"].map({
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall"
})

# Extract hotel_id and city from hotel_url
df["hotel_id"] = df["hotel_url"].str.extract(r"Reviews-(.*?)-")[0]
df["city"] = df["hotel_url"].str.extract(r"-Reviews-.*?-(.*?)(_Province|_State)")[0]


print(df[["author", "hotel_id", "city", "rating", "cleanliness", "service", "text"]].head())

df.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/raw/HotelDataset_Cleaned.csv", index=False)
