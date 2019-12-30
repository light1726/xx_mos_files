import numpy as np
import csv


def _get_order(order_f='order.txt'):
    order = []
    with open(order_f, 'r', encoding='utf-8') as f:
        line_cout = 0
        for line in f:
            if line_cout <= 4:
                line_cout += 1
                continue
            else:
                d = line.strip().split(' ')
                d = [int(i) for i in d]
                order.append(d)
    return np.array(order)


def _sort_list(lst, order):
    zipped_pairs = zip(order, lst)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


def _parse(lst, order):
    assert len(lst) == 100
    lst = np.array(lst)
    lst = lst.reshape((-1, 5))
    ordered_lst = []
    for i in range(lst.shape[0]):
        d = lst[i, :].tolist()
        sub_lst = _sort_list(d, order[i, :])
        # print('src: ', d)
        # print('order: ', order[i, :])
        # print('sorted:', sub_lst)
        ordered_lst += sub_lst
    return ordered_lst


def _get_statistics(arr, confidence=0.95, axis=0):
    z_dict = {0.8: 1.282, 0.85: 1.440, 0.9: 1.645, 0.95: 1.960,
              0.99: 2.576, 0.995: 2.807, 0.999: 3.291}
    z = z_dict[confidence]
    n = arr.shape[0]
    mu = np.mean(arr, axis=axis)
    sigma = np.std(arr, axis=axis)
    signifiance = z * sigma / np.sqrt(n)
    return mu, signifiance


def analyze(csv_file, mode='similarity'):
    """
    :param csv_file:
    :param mode: one of 'similarity' and 'naturalness'
    :return:
    """
    if mode not in ['similarity', 'naturalness']:
        raise ValueError("mode must be one of 'similarity' and 'naturalness'")
    lower_ind = 21 if mode is 'similarity' else 20
    upper_ind = 239 if mode is 'similarity' else 219
    order = _get_order()
    data = []
    with open(csv_file, encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                d = []
                for i in row[lower_ind: upper_ind]:  # similarity: [21:239], mos: [20:219]
                    if i != '':
                        d.append(int(i))
                if len(d) != 100:
                    print('Response {}: Not complete!'.format(row[0]))
                else:
                    data.append(d)
    data = np.array(data)
    sorted_data = []
    for i in range(data.shape[0]):
        sorted_data.append(_parse(data[i, :], order))
    print("{} subjects finished the survey in total!".format(len(sorted_data)))
    sorted_data = np.array(sorted_data)
    sorted_data = sorted_data.reshape((-1, 5))
    mu, significance = _get_statistics(sorted_data)
    print(mu)
    print(significance)
    return mu, significance


if __name__ == '__main__':
    csv_f = 'non_parallel_similarity_1228.csv'
    analyze(csv_f, mode='similarity')
