import pandas as pd
import hashlib
import os


input_file = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/raw/Hotel_Dataset_Dropped_Col.csv'
temp_file = '/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/raw/Hotel_Dataset_Without_duplicates.csv'

# Store row hashes
seen = set()


with open(temp_file, "w", encoding="utf-8", newline='') as f_out:
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=500_000)):
        deduped_rows = []

        for _, row in chunk.iterrows():
            row_hash = hashlib.md5(str(row.values).encode()).hexdigest()
            if row_hash not in seen:
                seen.add(row_hash)
                deduped_rows.append(row)

        if deduped_rows:
            df_chunk = pd.DataFrame(deduped_rows)
            df_chunk.to_csv(f_out, index=False, header=(i == 0))


os.remove(input_file)
