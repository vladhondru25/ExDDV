import sqlite3


class DatabaseConnector(object):
    def __init__(self, cfg=None) -> None:
        # Create a connection to the SQLite database
        # If the database file does not exist, it will be created
        self.connection = sqlite3.connect('my_database.db')

        # Create a cursor object to execute SQL queries
        self.cursor = self.connection.cursor()
        # Create a table (if it doesn't already exist)
        self.cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user VARCHAR(45) NOT NULL,
                video_name VARCHAR(100) NOT NULL,
                text MEDIUMTEXT NOT NULL,
                dataset VARCHAR(45) NOT NULL,
                manipulation VARCHAR(100) NOT NULL,
                click_locations TEXT
            )
            '''
        )

        # Commit the changes and close the connection
        self.connection.commit()
        print("Database and table created successfully!")

        self.insert_query = "INSERT INTO annotations (user, video_name, text, dataset, manipulation, click_locations) VALUES (?, ?, ?, ?, ?, ?)"
        self.select_query = "SELECT video_name FROM annotations WHERE user = '{}'"

        self.select_all_query = "SELECT * FROM annotations"

    def add_row(self, user, video_name, text, dataset, manipulation, click_locations):
        self.cursor.execute(self.insert_query, (user, video_name, text, dataset, manipulation, click_locations))
        self.connection.commit()

    def read_movie_entries(self, username):
        self.cursor.execute(self.select_query.format(username))
        rows = self.cursor.fetchall()
                
        return set(v[0] for v in rows)
    
    def read_all_movies(self):
        self.cursor.execute(self.select_all_query)
        rows = self.cursor.fetchall()

        return rows

    def close(self):
        self.connection.close()
