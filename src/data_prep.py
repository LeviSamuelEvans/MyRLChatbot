import os
from datasets import load_dataset
import pandas as pd

def load_data():
    dataset = load_dataset("daily_dialog")
    return dataset

def preprocess_data(dataset):
    # convert data to dataframe
    processed_data = {
        'train': pd.DataFrame(dataset['train']),
        'validation': pd.DataFrame(dataset['validation']),
        'test': pd.DataFrame(dataset['test'])
    }
    return processed_data

def save_data(data, data_dir, data_type):
    path = os.path.join(data_dir, f"{data_type}.csv")
    data.to_csv(path, index=False)
    print(f"Saved {data_type} data to {path}")

def main():
    raw_data_dir = os.path.join("data", "raw")
    processed_data_dir = os.path.join("data", "processed")

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # Load and save raw data
    dataset = load_data()
    for split in dataset.keys():
        dataset[split].to_csv(os.path.join(raw_data_dir, f"{split}.csv"))
        print(f"Saved raw {split} data to {os.path.join(raw_data_dir, f'{split}.csv')}")

    # no method for preprocessing yet, but there when we come to it..
    processed_data = preprocess_data(dataset)
    for split, data in processed_data.items():
        save_data(data, processed_data_dir, split)

    # quick examine of the data
    print("Sample processed training data:")
    print(processed_data['train'].head())

if __name__ == "__main__":
    main()
