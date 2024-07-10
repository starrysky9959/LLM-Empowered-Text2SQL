from typing import Dict, List, Set

from sql_metadata import Parser


class ColumnSchema:
    def __init__(
        self,
        column_name: str,
        column_type=None,
        column_description: str = None,
        value: str = None,
    ):
        self.column_name = column_name
        self.column_type = column_type
        self.column_description = column_description
        self.column_values = set()
        if value is not None:
            self.column_values.add(value)

    def add_sample_value(self, value: Set[str]):
        self.column_values.update(value)

    def __hash__(self):
        return hash(self.column_name)

    def __eq__(self, other):
        if isinstance(other, ColumnSchema):
            return self.column_name == other.column_name
        return False

    def __str__(self):
        return f"\tColumn(column_name={self.column_name}, column_type={self.column_type}, column_description={self.column_description}, sample_values={self.column_values})"


class TableSchema:
    def __init__(self, table_name=None):
        self.table_name = table_name
        # <column_name, ColumnSchema>
        self.columns = {}
        self.pk_names = []
        # (column_name, referr_out_table, refer_out_column)
        self.fk_out = []
        # (column_name, refer_in_table, refer_in_column)
        self.fk_in = []

    def contains(self, column_name: str, ignore_case: False):
        return column_name in self.columns

    def align(self, unalign_column_name: str):
        if unalign_column_name in self.columns:
            return unalign_column_name
        else:
            for column in self.columns.keys():
                if column.lower() == unalign_column_name.lower():
                    return column
            return None

    def add_column(self, column: ColumnSchema):
        if isinstance(column, ColumnSchema):
            if column.column_name in self.columns:
                self.columns[column.column_name].add_sample_value(column.column_values)
            else:
                self.columns[column.column_name] = column
        else:
            raise ValueError("column must be an instance of ColumnSchema")

    def merge(self, other):
        """
        always keep pk and fk in columns
        """
        # print("merge")
        # print(self.columns.keys())
        # print(other.columns.keys())
        self.pk_names = other.pk_names
        self.fk_out = other.fk_out
        self.fk_in = other.fk_in
        for column_name, column_obj in self.columns.items():
            column_obj.column_type = other.columns[column_name].column_type
            if column_obj.column_description is None:
                column_obj = other.columns[column_name].column_description
            if column_obj.column_values is None:
                column_obj = other.columns[column_name].column_values

        # add fk information
        for column_name in self.pk_names:
            if column_name not in self.columns:
                self.columns[column_name] = other.columns[column_name]
        for column_name, referr_out_table, refer_out_column in self.fk_out:
            if column_name not in self.columns:
                self.columns[column_name] = other.columns[column_name]
        for column_name, referr_in_table, refer_in_column in self.fk_in:
            if column_name not in self.columns:
                self.columns[column_name] = other.columns[column_name]

    def __str__(self):
        column_str = "\n".join([str(value) for value in self.columns.values()])
        pk_str = "\tPrimary Key:" + ",".join(self.pk_names)
        fk_str = ""
        for column_name, referr_out_table, refer_out_column in self.fk_out:
            fk_str += f"\tForeign Key:`{self.table_name}.{column_name}` reference `{referr_out_table}.{refer_out_column}`\n"
        return f"Table: {self.table_name}\n{column_str}\n{pk_str}\n{fk_str}"

    def get_column_names(self, with_table_name=True):
        column_names = self.columns.keys()
        if with_table_name:
            return [f"{self.table_name}.{col}" for col in column_names]
        else:
            return column_names


def prescsion_and_recall(A: set, B: set):
    intersection = A.intersection(B)

    if len(A) == 0:
        if len(B) == 0:
            precision = 1
        else:
            precision = 0
    else:
        precision = len(intersection) / len(A)

    if len(B) == 0:
        recall = 1
    else:
        recall = len(intersection) / len(B)

    return precision, recall


class DatabaseSchema:
    tables: Dict[str, TableSchema]

    def __init__(
        self,
        tables: Dict[str, TableSchema] = None,
        table_names: list = None,
    ):
        if tables is not None:
            self.tables = tables
        else:
            assert table_names is not None
            self.tables = dict()
            for table_name in table_names:
                self.tables[table_name] = TableSchema(table_name)

    def __getitem__(self, key):
        return self.tables[key]

    def __str__(self):
        return "".join([str(table) for table in self.tables.values()])

    def schema_size(self):
        table_count = len(self.tables)
        column_count = sum(map(lambda key: len(self.tables[key].columns), self.tables))
        return table_count, column_count

    def eval_precision_and_recall(self, refer):
        self_table_set = set(self.tables.keys())
        refer_table_set = set(refer.tables.keys())
        table_precision, table_recall = prescsion_and_recall(
            self_table_set, refer_table_set
        )
        self_column_set = set(self.get_column_names())
        refer_column_set = set(refer.get_column_names())
        column_precision, column_recall = prescsion_and_recall(
            self_column_set, refer_column_set
        )
        return table_precision, table_recall, column_precision, column_recall

    def merge(self, other):

        for table_name in self.tables:
            self.tables[table_name].merge(other.tables[table_name])

    def table_names(self):
        return self.tables.keys()

    def get_column_names(self):
        all_column_names = []
        for table in self.tables.values():
            all_column_names.extend(table.get_column_names())
        return all_column_names

    def search_column(self, column_name):
        ans = []
        for table in self.tables.values():
            if table.contains(column_name):
                ans.append(table.table_name)
        return ans

    def align_table(self, unalign_table_name: str):
        if unalign_table_name in self.tables:
            return unalign_table_name
        else:
            for table in self.tables.keys():
                if table.lower() == unalign_table_name.lower():
                    return table
            return None

    def align_column(self, table_name: str, unalign_column_name: str):
        table = self.tables[table_name]
        return table.align(unalign_column_name)

    def filter(self, sql: str):
        pass


# r = get_relevant_tables(
#     "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1"
# )
# print(r)
