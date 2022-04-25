import numpy as np

W = 10

def getBayesParams(data: np.ndarray):
    n, m = data.shape
    params = np.array([0.0] * (2 * W + 1))
    y1 = sum(x == 1 for x in data[:,-1])
    params[0] = (1.0 + y1 * m) / (2.0 + n * m)
    for k in range(1, W + 1):
        params[k] = (1 + sum(sum(x == k for x in row[:-1]) for row in filter(lambda x : x[-1] == 1, data))) / (W + m * y1)
        params[k + W] = (1 + sum(sum(x == k for x in row[:-1]) for row in filter(lambda x : x[-1] == 0, data))) / (W + m * (n - y1))
    return params

def bayesPredict(params: np.ndarray, row: np.ndarray):
    L = params[0] * np.prod([params[i] for i in row])
    R = (1.0 - params[0]) * np.prod([params[i + W] for i in row])
    return int(L / (L + R) >= .5)