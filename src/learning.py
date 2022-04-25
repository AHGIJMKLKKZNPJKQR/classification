import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from matplotlib import pyplot as plt
from numpy import array
import sys

if len(sys.argv) != 3:
    exit(1)

fig, ax = plt.subplots()
fig.set_size_inches(w = 5, h = 5)
ax.set_xlabel('Training set fraction')
ax.set_ylabel(f'Average error')

pos = { 'accuracy': 0, 'sensitivity': 1, 'precision': 2 }

x = []
ymin = []
yavg = []
ymax = []

file = open('out/bayes.learn', 'r')
for line in file.readlines():
    frac, arr = line.strip().split(None, 1)
    arr = eval(arr)
    x.append(float(frac))
    ymin.append(1 - arr[0][pos[sys.argv[2]]])
    yavg.append(1 - arr[1][pos[sys.argv[2]]])
    ymax.append(1 - arr[2][pos[sys.argv[2]]])
file.close()

ax.fill_between(x, ymin, ymax, alpha = 0.2, zorder = 1)
ax.plot(x, yavg, zorder = 2, label = 'Naive bayes classifier')

x = []
ymin = []
yavg = []
ymax = []

file = open('out/logistic.learn', 'r')
for line in file.readlines():
    frac, arr = line.strip().split(None, 1)
    arr = eval(arr)
    x.append(float(frac))
    ymin.append(1 - arr[0][pos[sys.argv[2]]])
    yavg.append(1 - arr[1][pos[sys.argv[2]]])
    ymax.append(1 - arr[2][pos[sys.argv[2]]])
file.close()

ax.fill_between(x, ymin, ymax, alpha = 0.2, zorder = 1, color = 'red')
ax.plot(x, yavg, zorder = 2, color = 'red', label = 'Logistic regression')
ax.legend()

# plt.show()
plt.savefig(f'learning-{sys.argv[2]}.pgf', format="pgf")

