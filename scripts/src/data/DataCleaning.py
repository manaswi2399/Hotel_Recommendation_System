from dask.distributed import Client
import dask.dataframe as dd
import os

def main():
    client = Client(n_workers=8, threads_per_worker=2, memory_limit='8GB')
    print("Dask client started")

    # Load data
    df = dd.read_csv('/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/raw/HotelDataset.csv', parse_dates=['date'], low_memory=False)

    # Drop unwanted columns
    cols_to_drop = [
        'ur_question.prompt.11',
        'userrating.prompt.190',
        'userrating.prompt.46',
        'userrating.prompt.48',
        'business service (e.g., internet access)',
        'check in / front desk',
    ]
    df = df.drop(columns=cols_to_drop)
    output_path = "/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/raw/Hotel_Dataset_Dropped_Col.csv"
    df.to_csv(output_path, single_file=True, index=False)


    os.remove("/home/018171153/Hotel_Recommendation_System/Hotel_Recommendation_System/scripts/src/data/raw/HotelDataset.csv")
    print("Old file removed and new file saved.")

if __name__ == "__main__":
    main()
    