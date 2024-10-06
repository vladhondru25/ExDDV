import pandas as pd

from database import DatabaseConnector


if __name__ == "__main__":
    rows = []
    for db_name in ["eduard_database.db", "vlad_database.db"]:
        db = DatabaseConnector(db_name=db_name)
        rows.extend(db.read_all_movies())

    df = pd.DataFrame(
        rows,
        columns=["id", "username", "movie_name", "text", "dataset", "manipulation", "click_locations"]
    )

    df.to_csv("dataset.csv", encoding='utf-8', index=False)
