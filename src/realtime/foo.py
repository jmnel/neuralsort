import pickle
from pprint import pprint

with open('data/AppDownloadSP500_data.180604_140339.p', 'rb') as infile:

    new_dict = pickle.load(infile)

print(new_dict.keys())
# pprint(new_dict['tickReqParams'])
# pprint(new_dict['tickSize'])
# print(new_dict['tickPrice'][0].keys())

ticks = new_dict['tickPrice']

# for idx in range(len(ticks)):
#    pprint(ticks[idx])

# print(len(ticks))

TICK_DELAYED_BID = 66
TICK_DELAYED_ASK = 67
TICK_DELAYED_LAST = 68
TICK_DELAYED_BID_SIZE = 69
TICK_DELAYED_ASK_SIZE = 70
TICK_DELAYED_LAST_SIZE = 71
TICK_DELAYED_HIGH = 72
TICK_DELAYED_LOW = 73
TICK_DELAYED_VOLUME = 74
TICK_DELAYED_CLOSE = 75
TICK_DELAYED_OPEN = 76
# TICK_TYPES = {66: 'DELAYED_BID',
#              67: 'DELAYED_ASK',
#              68: 'DELAYED_LAST',
#              69: 'DELAYED_BID_SIZE',
#              70: 'DELAYED_ASK_SIZE',
#              71: 'DELAYED_LAST_SIZE',
#              72: 'DELAYED_HIGH',
#              73: 'DELAYED_LOW',
#              75: 'DELAYED_CLOSE',
#              74: 'DELAYED_VOLUME',
#              76: 'DELAYED_OPEN'}


req_ids = {t['reqId'] for t in ticks}
# pprint(len(req_ids))

#tick_types = {t['tickType'] for t in ticks}
# print(tick_types)
# pprint(new_dict)
