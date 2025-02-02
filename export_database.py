import os
import uuid

import pandas as pd

from database import DatabaseConnector


DATABASES_INPUT_PATH = "/home/vhondru/vhondru/phd/biodeep/xAI_deepfake/databases"


if __name__ == "__main__":
    rows = []
    for db_name in os.listdir(DATABASES_INPUT_PATH):
        db = DatabaseConnector(db_name=os.path.join(DATABASES_INPUT_PATH,db_name))
        rows.extend(db.read_all_movies())

    df = pd.DataFrame(
        rows,
        columns=["id", "username", "movie_name", "text", "dataset", "manipulation", "click_locations", "difficulty"]
    )

    # Filter out clicked locations
    print(f'Before: {len(df)}')
    df = df[df.click_locations != r"{}"]
    print(f'After: {len(df)}')

    df["id"] = range(len(df))

    df.to_csv("/home/vhondru/vhondru/phd/biodeep/xAI_deepfake/dataset.csv", encoding='utf-8', index=False)
