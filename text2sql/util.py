class AverageCalculator:
    def __init__(self):
        self.values = []

    def add_value(self, value):
        self.values.append(value)

    def calculate_average(self):
        if not self.values:
            return 0
        return sum(self.values) / len(self.values)


class SchemaSizeAverageCalculator:
    def __init__(self):
        self.table_count = AverageCalculator()
        self.column_count = AverageCalculator()

    def add_value(self, value_tuple):
        # print(value_tuple)
        self.table_count.add_value(value_tuple[0])
        self.column_count.add_value(value_tuple[1])

    def calculate_average(self):
        return (
            self.table_count.calculate_average(),
            self.column_count.calculate_average(),
        )


class PrecisionRecallAverageCalculator:
    def __init__(self):
        self.table_precision = AverageCalculator()
        self.table_recall = AverageCalculator()
        self.column_precision = AverageCalculator()
        self.column_recall = AverageCalculator()

    def add_value(self, value_tuple):
        print(value_tuple)
        self.table_precision.add_value(value_tuple[0])
        self.table_recall.add_value(value_tuple[1])
        self.column_precision.add_value(value_tuple[2])
        self.column_recall.add_value(value_tuple[3])

    def calculate_average(self):
        return (
            self.table_precision.calculate_average(),
            self.table_recall.calculate_average(),
            self.column_precision.calculate_average(),
            self.column_recall.calculate_average(),
        )
