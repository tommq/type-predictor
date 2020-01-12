from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
money = [75, 93, 88, 84, 81]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.0f' % (x)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.bar(x, money)
plt.xticks(x, ('LR', 'RF', 'SGD', 'MLP', 'SVC'))
plt.show()