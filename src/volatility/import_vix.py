from requests import request
from pprint import pprint
import pandas as pd
import pandasgui
import io
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
import scipy.stats as stats
import numpy as np

CBOE_VIX_2014_URL = "http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv"
CBOE_VIX_1990_URL = "http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixarchive.xls"

QUANDL_DATABASE_PATH = settings.


def prepare_database():
    pass


def download():

    vix_csv_2014 = io.BytesIO(request('get',
                                      CBOE_VIX_2014_URL).content)

    vix_xls_1990 = io.BytesIO(request('get',
                                      CBOE_VIX_1990_URL).content)

    vix_2014_df = pd.read_csv(vix_csv_2014, header=1)
    vix_1990_df = pd.read_excel(vix_xls_1990, header=1)

    vix_2014_df.Date = [datetime.strptime(
        d, '%m/%d/%Y') for d in vix_2014_df.Date]

    vix_1990 = vix_1990_df.to_numpy()

    vix_1990[:, 0] = [d.to_pydatetime().date() for d in vix_1990[:, 0]]
    vix_2014 = vix_2014_df.to_numpy()
    vix_2014[:, 0] = [d.to_pydatetime().date() for d in vix_2014[:, 0]]

    vix = np.concatenate([vix_1990, vix_2014], axis=0)

    for t in range(len(vix)):
        if np.isnan(vix[t, 4]):
            assert(t > 0 and t + 1 < len(vix))
            vix[t, 4] = 0.5 * (vix[t - 1, 4] + vix[t + 1, 4])

    plt.plot(vix_1990[:, 0], vix_1990[:, 4], lw=0.10)
    plt.plot(vix_2014[:, 0], vix_2014[:, 4], lw=0.10)

    gamma = 2
    l = 50
    ema = np.zeros(len(vix))

    print(len(ema))
    ema[0] = vix[0, 4]
    for t in range(1, len(ema)):
        print(t)
        assert(np.isnan(vix[t, 4]) == False)
        ema[t] = vix[t, 4] * gamma / \
            (1 + l) + ema[t - 1] * (1 - gamma / (1 + l))

    print(len(vix[:, 0]))

    pprint(ema)

    plt.plot(vix[:, 0], ema, lw=0.8)
    plt.show()


def main():

    vix_2014 = download()


main()
