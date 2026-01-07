import pandas as pd
from config.settings import KAGGEL_DATASET
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import ast
from database.operations import insert_raw_articles

def load_kaggle_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load and preprocess the Kaggle Medium articles dataset.

    Args:
        dataset_path (Path): Path to the CSV dataset file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # 1️⃣ Check if file exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # 2️⃣ Load CSV
    df = pd.read_csv(dataset_path)

    # 3️⃣ Quick inspection (optional prints)
    print("Data types:\n", df.dtypes)
    print("\nInfo:")
    print(df.info())
    print("\nMissing values per column:\n", df.isna().sum())

    # 4️⃣ Clean duplicates and missing required fields
    df = df.drop_duplicates()
    df = df.dropna(subset=['url', 'tags'])

    # 5️⃣ Clean and convert columns
    df['title'] = df['title'].fillna("").astype(str)
    df['body'] = df['text'].fillna("").astype(str)
    df['url'] = df['url'].astype(str)
    df['source'] = "kaggle"
    
    # 6️⃣ Convert tags column to a joined string
    df['category'] = df['tags'].apply(
        lambda x: "|".join(ast.literal_eval(x)) if pd.notna(x) else ""
    )

    # 7️⃣ Drop unneeded columns
    df = df.drop(columns=['text', 'authors', 'timestamp', 'tags'])

    # 8️⃣ Optional prints
    print("\nProcessed dtypes:\n", df.dtypes)
    print("\nSample rows:\n", df.head(5))

    return df


def main():
    df=load_kaggle_dataset(KAGGEL_DATASET)
    for row in df.itertuples(index=False,name=None):
        row_dict=dict(zip(df.columns,row))
        insert_raw_articles(row_dict)

if __name__=="__main__":
    main()