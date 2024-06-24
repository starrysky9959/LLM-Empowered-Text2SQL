from typing import Set


class ColumnSchema:
    def __init__(
        self,
        column_name: str,
        column_type,
        column_description: str = None,
        value: str = None,
    ):
        self.column_name = column_name
        self.column_type = column_type
        self.column_description = column_description
        self.sample_values = set()
        if value is not None:
            self.sample_values.add(value)

    def add_sample_value(self, value: Set[str]):
        self.sample_values.update(value)

    def __hash__(self):
        return hash(self.column_name)

    def __eq__(self, other):
        if isinstance(other, ColumnSchema):
            return self.column_name == other.column_name
        return False

    def __str__(self):
        return f"\tColumn(column_name={self.column_name}, column_type={self.column_type}, column_description={self.column_description}, sample_values={self.sample_values})"


class TableSchema:
    def __init__(self, table_name=None):
        self.table_name = table_name
        # <column_name, ColumnSchema>
        self.columns = {}
        self.pk_names = []
        self.fk_names = []

    def add_column(self, column: ColumnSchema):
        if isinstance(column, ColumnSchema):
            if column.column_name in self.columns:
                self.columns[column.column_name].add_sample_value(column.sample_values)
            else:
                self.columns[column.column_name] = column
        else:
            raise ValueError("column must be an instance of ColumnSchema")

    def merge(self, other):
        print(f"other:{other}")
        print(other.columns.keys())
        self.pk_names = other.pk_names
        self.fk_names = other.fk_names
        for column_name in self.columns:
            self.columns[column_name].column_type = other.columns[column_name].column_type

    def __str__(self):
        column_str = "\n".join([str(value) for value in self.columns.values()])
        return f"Table: {self.table_name}\n {column_str}"
