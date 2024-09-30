import pandas as pd

from database import DatabaseConnector


if __name__ == "__main__":
    db = DatabaseConnector()
    rows = db.read_all_movies()

    df = pd.DataFrame(
        rows,
        columns=["id", "username", "movie_name", "text", "dataset", "manipulation", "click_locations"]
    )

    df.to_csv("dataset.csv", encoding='utf-8', index=False)
