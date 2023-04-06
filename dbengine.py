
class DBEngine:
    def __init__(self, connector) -> None:
        # Initialize the DBEngine with a connector
        self.connector = connector

    def preprocessing(self):
        # Preprocess the data, including create samples, create pre-aggregations, create indexes, etc.
        pass

    def execute(self, query, params=None):
        # Execute a query and return the result
        pass

    def execute_approx(self, query, params=None):
        # Execute a query with approximations and return the result
        pass

    def query_rewriting(self, query):
        # Rewrite the query
        pass

    def approx_rewriting(self, query):
        # Rewrite the query with approximations
        pass

    def update(self, query):
        # Update the data
        pass

    def append(self, query):
        # Append the data
        pass


class ClickHouseEngine(DBEngine):
    def __init__(self, connector) -> None:
        super().__init__(connector)

    def preprocessing(self):
        # Preprocess the data, including create samples, create pre-aggregations, create indexes, etc.
        pass

    def execute(self, query, params=None):
        # Execute a query and return the result
        with self.connector.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    def execute_approx(self, query, params=None):
        # Execute a query with approximations and return the result
        query = self.approx_rewriting(query)
        return self.execute(query, params)

    def query_rewriting(self, query):
        # Rewrite the query
        pass

    def approx_rewriting(self, query):
        # Rewrite the query with approximations
        pass

    def update(self, query):
        # Update the data
        pass

    def append(self, query):
        # Append the data
        pass
