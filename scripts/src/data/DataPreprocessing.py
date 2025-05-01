import pandas as pd

# Load dataset
df = pd.read_csv("/home/018171153/Hotel_Recommendation_System/scripts/src/data/raw/Hotel_Dataset_Without_duplicates.csv")

# Drop rows with missing critical fields
df.dropna(subset=["author", "rating", "text", "hotel_url"], inplace=True)

# Clean string fields
df["author"] = df["author"].str.strip()
df["title"] = df["title"].fillna("").str.strip().str.lower()
df["text"] = df["text"].str.strip().str.lower()

# Rename location if needed
if "location" in df.columns:
    df.rename(columns={"location": "location_rating"}, inplace=True)

# Handle sub-ratings and fill missing values hotel-wise
"""sub_rating_cols = ["cleanliness", "service", "rooms", "sleep quality", "value", "location_rating"]
for col in sub_rating_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df.groupby("hotel_url")[col].transform("mean"))"""

# Clean and convert 'date' by stripping timestamp and parsing
df["date"] = df["date"].astype(str).str[:10]
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Extract month, year, season
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["season"] = df["month"].map({
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall"
})

# Extract hotel_id from hotel_url
df["hotel_id"] = df["hotel_url"].str.extract(r"Reviews-(.*?)-")[0]


df.to_csv("/home/018171153/Hotel_Recommendation_System/scripts/src/data/raw/HotelDataset_CleanedBeforeSplit.csv", index=False)
