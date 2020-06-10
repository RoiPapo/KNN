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


class MinMax:
    def __init__(self):
        self.min_list = []
        self.max_list = []

    def fit(self, points):
        for point in points:
            self.max_list.append(max(point.coordinates))
            self.min_list.append(min(point.coordinates))

    def transform(self, points):
        new = []
        for i, p in enumerate(points):
            new_coordinates = p.coordinates
            for cordinate in new_coordinates:
                cordinate = (cordinate - self.min_list[i]) / (self.max_list[i] - self.min_list[i])
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class SumNormalizer:
    def __init__(self):
        self.normalized_list = []

    def fit(self, points):
        pass

    def transform(self, points):
        coordinate_sum = 0
        for point in points:
            for coordinate in point.coordinates:
                coordinate_sum += abs(coordinate)
            point.coordinates = coordinate_sum
            self.normalized_list.append(point)
            coordinate_sum = 0
        return self.normalized_list
