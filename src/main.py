import numpy as np

from parse import getStandardizationParams, parseData, standardize
from bayes import getBayesParams, bayesPredict
from logistic import getLogisticParams, logisticPredict

trainingSizes = np.array([.01, .02, .03, .125, .25, .375, .5, .625, .75, .875, 1.0])
RUNS = 75
bayesFile = open("out/bayes.learn", 'w')
logisticFile = open("out/logistic.learn", 'w')

def runSingle(frac, getParams, predict, std = False):
    training, test = parseData()
    if std == True:
        stdParams = getStandardizationParams(training)
        training = standardize(training, stdParams)
        test = standardize(test, stdParams)
    training = training[:int(len(training) * frac)]

    params = getParams(training)

    br = np.array([[0, 0], [0, 0]])
    for row in test:
        br[int(predict(params, row[:-1]))][int(row[-1])] += 1
    return (br[0][0] + br[1][1]) / br.sum(), br[1][1] / (br[1][1] + br[0][1]), br[1][1] / (br[1][1] + br[1][0])

def run(runs: int, frac, getParams, predict, file, std = False):
    print(f"Fraction: {frac}")
    avg = np.zeros(3)
    low = np.ones(3)
    high = np.zeros(3)
    for _ in range(runs):
        res = runSingle(frac, getParams, predict, std)
        avg += res
        for i in range(3):
            low[i] = min(res[i], low[i])
            high[i] = max(res[i], high[i])

    avg /= runs
    print(f"AVERAGE: Accuracy: {avg[0]}, Sensitivity: {avg[1]}, Precision: {avg[2]}, F1: {2 / (1 / avg[1] + 1 / avg[2])}")
    file.write(str(frac) + "\t" + repr(np.array([low, avg, high])).replace('\n', '') + "\n")

def runBayes(runs: int, frac):
    run(runs, frac, getBayesParams, bayesPredict, bayesFile)

def runLogistic(runs: int, frac):
    run(runs, frac, getLogisticParams, logisticPredict, logisticFile, True)

def doBayes():
    print("DOING BAYES")
    for size in trainingSizes:
        runBayes(RUNS, size)

def doLogistic():
    print("DOING LOGISTIC")
    for size in trainingSizes:
        runLogistic(RUNS, size)

doBayes()
bayesFile.close()
doLogistic()
logisticFile.close()

