import numpy as np
from itertools import product

def exp_cum_dev(avg, loc):
    return 1 / (1 - np.exp(-loc/avg))

def generate_onepertriangle(size, tail_perc=0.01, ultimate_mean=1000, ultimate_std=200):
    beta = - (size) / np.log(tail_perc)

    index = np.arange(size) + 1
    cdfs = np.array(list(map(lambda x: exp_cum_dev(beta, x), index)))
    perc_dev = np.insert(1 / cdfs, 0, 0, 0)
    perc = np.diff(perc_dev)
    ultimates = np.random.normal(ultimate_mean, ultimate_std, (1, size))

    one_per_triangle = np.matmul(perc.reshape((size, 1)), ultimates).transpose()
    for i, j in product(range(size), range(size)):
        if i + j >= size:
            one_per_triangle[i, j] = 0
    return one_per_triangle

if __name__ == '__main__':
    np.random.seed(2)
    tri_information = generate_onepertriangle(size=15, tail_perc=0.01, ultimate_mean=100000, ultimate_std=5000)
    print(tri_information)
    #np.savetxt("test_gen_triangle.csv", tri_information[0], delimiter=";", fmt='%10.0f')