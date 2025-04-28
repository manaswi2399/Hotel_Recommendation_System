import pandas as pd

def proportional_sample(df, year_col, target_total, seed=42):
    sampled_frames = []
    years = sorted(df[year_col].dropna().unique())
    year_counts = df[year_col].value_counts(normalize=True)
    
    for year in years:
        year_df = df[df[year_col] == year]
        n_samples = int(target_total * year_counts.get(year, 0))
        n_samples = min(n_samples, len(year_df))  # Safety check
        if n_samples > 0:
            sampled_frames.append(year_df.sample(n=n_samples, random_state=seed))
    
    final_sample = pd.concat(sampled_frames)
    return final_sample


# Paths
train_path = "/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_with_topics.csv"
val_path = "/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Validation_with_topics.csv"
test_path = "/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Test_with_topics.csv"


train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)


train_small = proportional_sample(train_df, "year", target_total=5_000_000)
val_small = proportional_sample(val_df, "year", target_total=1_000_000)
test_small = proportional_sample(test_df, "year", target_total=1_000_000)

train_small.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Train_small.csv", index=False)
val_small.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Validation_small.csv", index=False)
test_small.to_csv("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/data/processed/Test_small.csv", index=False)

print("\n Sampling done and CSVs saved!")
