from pprint import pprint
import random
from datetime import datetime, time, timedelta

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

# from tick_bar_dataset import TickBarDataset
# from iex_ticks_dataset import TickDatasetIEX
from iex_tick_bar_ds import TickBarDatasetIEX


class ActionSpace:

    def sample(self):
        #        return random.randint(-1, 1)
        return random.randint(0, 2)


class StateSpace:

    def __init__(self, state_space):
        self.shape = [state_space, ]


class TickEnvironment:

    def __init__(self, trade_penalty=0.05, invalid_penalty=0.1):

        self.train_loader = DataLoader(TickBarDatasetIEX(mode='trian'),
                                       batch_size=1,
                                       shuffle=True)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
#        self.init_len = 50
        self.trade_penalty = trade_penalty

        self.action_space = ActionSpace()
        self.state_space = StateSpace(50)

        self.symbol, self.day = None, None

        self.seq_len = 200

    def reset(self):
        """
        Reset the environment.

        """

#        while True:
        try:
            self.day, self.symbol, self.ticks = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            self.day, self.symbol, self.ticks = next(self.train_iter)
#            except ValueError:
#                self.day, self.symbol, self.ticks = next(self.train_iter)
#                print('value error')
#                exit()
#            except TypeError:
#                self.day, self.symbol, self.ticks = next(self.train_iter)
#                print('type error')
#                exit()
#            else:
#                break

        self.prices = self.ticks[0, :, 4]

        # Rescale tick timestamp using open and close timing of regular trading hours.
        t0 = self.ticks[0, 0, 0]
        t1 = self.ticks[0, -1, 0]
        t_open = datetime.combine(datetime.fromtimestamp(t0.item()), time(9, 30)).timestamp()
        t_close = datetime.combine(datetime.fromtimestamp(t0.item()), time(16)).timestamp()
        self.ticks[0, :, 0] = (self.ticks[0, :, 0] - t_open) / (t_close - t_open)

        self.cash = 1.0
        self.hold = 0.0

        self.idx = 0

        self.buy_pts = list()
        self.sell_pts = list()
        self.hold_intervals = list()
        self.flat_intervals = list()
        self.last_switch = 0

        self.actions = [0, ]
        self.rewards = [0, ]
        self.cash_hist = [self.cash, ]
        self.hold_hist = [self.hold, ]

        self.raw_short_count = 0
        self.raw_neutral_count = 0
        self.raw_long_count = 0

        self.net = 0

        a = torch.from_numpy(np.dstack((self.actions,
                                        self.rewards,
                                        self.cash_hist,
                                        self.hold_hist))).float()

        state = torch.cat((self.ticks[:, :self.idx + 1, :], a), dim=-1)
        pad_len = self.seq_len - state.shape[1]
        padding = torch.zeros((1, pad_len, state.shape[2]))

        state = torch.cat((padding, state), dim=1)

        self.prev_value = self.cash

        return state

    def step(self, action):

        #        print(action)

        # Current price is tick bar closing price.
        price = self.ticks[0, self.idx, 4]

        done = False
        invalid = False

        reward = 0

        # Short action.
        if action == 0:
            self.raw_short_count += 1
            if self.hold <= 0.0:
                done = True
                invalid = True
            else:
                self.cash = self.hold * price - self.trade_penalty
                self.hold = 0
                self.sell_pts.append((self.idx, price))
#                self.hold_intervals.append((

        # Neutral action.
        elif action == 1:
            self.raw_neutral_count += 1
            reward -= 1e-2

        # Long position.
        elif action == 2:
            self.raw_long_count += 1
            if self.cash <= 0 or self.hold > 0:
                done = True
                invalid = True
            else:
                self.hold = (self.cash - self.trade_penalty) / price
                self.cash = 0
                self.buy_pts.append((self.idx, price))

        new_value = self.cash
        if self.hold > 0:
            new_value += self.hold * price - self.trade_penalty

        delta = new_value - self.prev_value

        if not invalid:
            reward += delta
#        print(delta)

#        reward += 1 * self.idx

        self.idx += 1
        self.actions.append(action)
        self.cash_hist.append(self.cash)
        self.hold_hist.append(self.hold)
        self.rewards.append(reward)

        if not done:
            done = self.idx + 1 >= self.prices.shape[0]

#        state_env = self.ticks[:, :self.idx + 1, :]
#        state_agent = torch.FloatTensor((self.cash, self.hold))
#        state = (state_env, state_agent)
        a = torch.from_numpy(np.dstack((self.actions,
                                        self.rewards,
                                        self.cash_hist,
                                        self.hold_hist))).float()

        state = torch.cat((self.ticks[:, :self.idx + 1, :], a), dim=-1)
        pad_len = self.seq_len - state.shape[1]

        assert state.shape[1] > 0

        if pad_len > 0:
            padding = torch.zeros((1, pad_len, state.shape[2]))
            state = torch.cat((padding, state), dim=1)

        else:
            state = state[:, -self.seq_len:]

        assert state.shape[1] == self.seq_len

        self.net = self.cash
        if self.hold > 0:
            self.net += self.hold * price - self.trade_penalty

        if done:
            #            reward += 1e-2 * (self.idx / self.ticks.shape[1])
            reward += 1e-2 * self.idx

        return state, reward, done

#        if done and action == 0:
#            self.flat_intervals.append((self.last_switch, self.idx))
#        if done and action == 1:
#            self.hold_intervals.append((self.last_switch, self.idx))

#        n = self.state_space.shape[0]
#        p = self.data[self.idx: self.idx + n]

#        p = np.diff(np.log(self.prices[self.idx - self.init_len: self.idx + 1])) * 10.0
#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1] / self.prices[0]
#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1]
#        state = p
#        state = np.concatenate((p, (action,)))
#        return state, reward, done

    def __len__(self):
        return self.prices.shape[0]


#env = TickEnvironment()
#s = env.reset()

# for idx in range(10):
#    state, reward, done = env.step(1)
#    print(f'{idx} -> {state.shape}')
#    print(state)
