from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1) ** 0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class MinMaxNormalizer:
    def __init__(self):
        self.min_list = []
        self.max_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.max_list = []
        self.min_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.min_list.append(min(values))
            self.max_list.append(max(values))

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.min_list[i]) / (self.max_list[i] - self.min_list[i]) for i in
                range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class SumNormalizer:
    def __init__(self):
        self.sum_of_coordinates = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        for i in range(len(all_coordinates[0])):
            value = sum([abs(x[i]) for x in all_coordinates])
            self.sum_of_coordinates.append(value)

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [new_coordinates[i] / self.sum_of_coordinates[i] for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new
