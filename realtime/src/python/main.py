import csv
import io
from datetime import datetime
from pprint import pprint
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn


def main():
    data = list()
    symbols = set()
    with open('../../data/2020-06-04.csv', 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)

    ts = list(filter(lambda r: r[1] == 'GOOGL', data))
#    print(ts[0])
#    exit()

    ts = list((int(r[0]), r[1], float(r[2]), int(r[3])) for r in ts)

    t, _, s, _ = zip(*ts)

    plt.plot(t, s, linewidth=0.4)
    plt.show()


if __name__ == '__main__':
    main()
