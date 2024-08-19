import mysql.connector
from mysql.connector import Error


class DatabaseConnector(object):
    def __init__(self, cfg) -> None:
        try:
            self.connection = mysql.connector.connect(
                host=cfg["DB_CREDENTIALS"]["HOST"],
                user=cfg["DB_CREDENTIALS"]["USER"],
                password=cfg["DB_CREDENTIALS"]["PASSWORD"],
                database=cfg["DB_CREDENTIALS"]["DATABASE"]
            )

            self.insert_query = "INSERT INTO annotations (user, video_name, text, dataset, manipulation) VALUES (%s, %s, %s, %s, %s)"
            self.select_query = "SELECT video_name FROM annotations WHERE user = '{}'"
            
            if not self.connection.is_connected():
                raise ValueError("Could not connect to ")
            
        except Error as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")

    def add_row(self, user, video_name, text, dataset, manipulation):
        cursor = self.connection.cursor()

        cursor.execute(self.insert_query, (user, video_name, text, dataset, manipulation))
        self.connection.commit()

        cursor.close()

    def read_movie_entries(self, username):
        with self.connection.cursor() as cursor:
            cursor.execute(self.select_query.format(username))
            rows = cursor.fetchall()
                
        return set(v[0] for v in rows)

    def close(self):
        if self.connection.is_connected():
            self.connection.close()
