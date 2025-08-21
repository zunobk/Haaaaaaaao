import pymysql
import re
import ast

class Database():
    def __init__(self):
        self.connection = self.create_connection()

    def create_connection(self):
        try:
            host_name       = "localhost"
            user_name       = "root"
            user_password   = "1018"
            db_name         = "videostates"

            connection = pymysql.connect(
                host=host_name,
                user=user_name,
                password=user_password,
                database=db_name
            )
            print("Connection to MySQL DB successful")
        except Exception as e:
            print(f"The error '{e}' occurred")
        return connection

    def execute_query(self, query):
        if not self.connection:
            raise Exception("No database connection")
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Exception as e:
            print(f"The error '{e}' occurred")

    def execute_read_query(self, query):
        if not self.connection:
            raise Exception("No database connection")
        cursor = self.connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"The error '{e}' occurred")

    def select_video_state(self, video_name):
        select_users_query = f"SELECT videostate from states where videoname='{video_name}';"
        tuple_str = self.execute_read_query(select_users_query)
        json_str = tuple_str[0][0]
        processed_str = re.sub(r'\\', r'/', json_str)
        video_state = ast.literal_eval(processed_str)

        return video_state

        


    


