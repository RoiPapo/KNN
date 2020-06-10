from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def create_classifiers(k, points):
    count = 1
    list_of_classifiers = []
    while count < 40:
        list_of_classifiers.append(0)
        count += 1

    score_dict = {}
    for i in range(k):
        print("for K number:", i + 1)
        list_of_classifiers[i + 1] = (KNN(i + 1, 2))
        list_of_classifiers[i + 1].train(points)
        cv = CrossValidation()
        score_dict[i + 1] = cv.run_cv(points, 10, list_of_classifiers[i + 1], accuracy_score)

    print(score_dict)


def question3(list_of_foldnums, cv, points, m, k, print_final_score=False):
    print('K=%d' % k)
    for i in range(len(list_of_foldnums)):
        temp = []
        print('%d-fold-cross-validation:' % list_of_foldnums[i])
        # print(list_of_foldnums[i],"-fold-cross-validation:")
        temp.append(cv.run_cv(points, list_of_foldnums[i], m, accuracy_score, print_final_score))


def test_Folds(list_of_foldnums, cv, points, m, norm, print_final_score=False):
    for i in range(len(list_of_foldnums)):
        accuracy = []
        accuracy.append(cv.run_cv(points, list_of_foldnums[i], m, accuracy_score, print_final_score))
        print("Accuracy of", norm.__name__, "is", accuracy[0])
        print("")


def normalized_check(points, k_list, normalize_method_list):
    for k in k_list:
        print('K=%d' % k)
        m = KNN(k)
        for norm in normalize_method_list:
            norm_obj = norm()
            norm_obj.fit(points)
            normelized_points = norm_obj.transform(points)
            cv = CrossValidation()
            test_Folds([2], cv, normelized_points, m, norm, print_final_score=False)


def run_knn(points):
    m = KNN(19)
    m.train(points)
    cv = CrossValidation()
    print("Question 3:")
    question3([2, 10, 20], cv, points, m, 7, print_final_score=False)
    print("Question 4:")
    normalized_check(points, [5, 7], [DummyNormalizer, SumNormalizer, MinMaxNormalizer, ZNormalizer])


if __name__ == '__main__':
    loaded_points = load_data()
    run_knn(loaded_points)
