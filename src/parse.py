import numpy as np

FILENAME = "data/rp.data"

TRAINING_FRAC = 2 / 3

def parseData():
    file = open(FILENAME, 'r')
    data = np.array([[int(x) for x in l.strip().split()] for l in list(file)])
    file.close()

    target = [0, 0]
    for row in data:
        row[-1] = int(row[-1] == 4)
        target[row[-1]] += 1
    np.random.default_rng().shuffle(data)
    training = []
    test = []
    count = [0, 0]
    for row in data:
        training.append(row) if count[row[-1]] < TRAINING_FRAC * target[row[-1]] else test.append(row)
        count[row[-1]] += 1
    return np.array(training), np.array(test)

def getStandardizationParams(data):
    n, m = data.shape
    avg = np.sum(data[:,:-1], 0) / n
    stddev = np.sqrt(np.sum(np.array([data[:, i] - avg[i] for i in range(m - 1)]) ** 2, 1) / n)
    return avg, stddev

def standardize(data, params):
    avg, stddev = params
    n, m = data.shape
    
    res = np.array(data, float)

    for i in range(n):
        for j in range(m - 1):
            res[i][j] = (data[i][j] - avg[j]) / stddev[j]
    return res