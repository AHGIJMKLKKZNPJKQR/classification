import numpy as np

EPS = .2

def h(v, x):
    return 1 / (1 + np.exp(-np.dot(v,  x)))

def getLogisticParams(data: np.ndarray):
    alpha = .3
    n, m = data.shape
    theta = np.zeros(m - 1)
    oldchange = 0.0
    for seed in range(n):
        maxchange = 0.0
        np.random.default_rng(seed).shuffle(data)

        for row in data:
            y = row[-1]
            x = row[:-1]
            delta = (y - h(theta, x)) * x
            theta += alpha * delta
            change = np.dot(delta, delta)
            if change > maxchange:
                maxchange = change

        if maxchange > oldchange:
            alpha *= 1 - EPS
        else:
            alpha *= 1 + EPS
        oldchange = maxchange
    return theta

def logisticPredict(params: np.ndarray, row: np.ndarray):
    return h(params, row) >= 0.5