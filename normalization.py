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
    def transform(self, points):
        self.list_for_normalize = points
        v = self.list_for_normalize[:, 1]  # foo[:, -1] for the last column
        self.list_for_normalize[:, 1] = (v - v.min()) / (v.max() - v.min())
        return self.list_for_normalize


class SumNormalizer:
    # def __init__(self):
    #     self.normalized_list = []

    def sum_normalizing(self, points):
        coordinate_sum = 0
        normalized_list = []
        for point in points:
            for coordinate in point.coordinates:
                coordinate_sum += abs(coordinate)
            normalized_list.append(coordinate_sum)
            coordinate_sum = 0
        return normalized_list
