from iex_parser import Parser, DEEP_1_0, TOPS_1_6
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

#f = 'data_feeds_20180127_20180127_IEXTP1_TOPS1.6.pcap.gz'
f = 'data_feeds_20190814_20190814_IEXTP1_TOPS1.6.pcap.gz'

p = list()
a = list()

n = 0

symbols = list()
with Parser(f, TOPS_1_6) as reader:
    for message in reader:

        if message['type'] == 'trade_report':
            print(message)
            symbol = message['symbol'].decode('utf-8')
            symbols.append((symbol, float(message['price'])))
        n += 1

#        if n % 1000 == 0:
#            print(n)
