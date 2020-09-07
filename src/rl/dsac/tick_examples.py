import sqlite3
from pprint import pprint

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import settings

IEX_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME


class TickExamples(Dataset):

    def __init__(self,
                 mode='train',
                 seq_length=20):
        super().__init__()

        self.mode = mode
        self.seq_length = 20

        self.split = (0.8, 0.1, 0.1)

        db = sqlite3.connect(IEX_PATH)

        rows = db.execute('''
SELECT day, symbol, message_count FROM iex_ticks_meta
WHERE message_count >= 2000;''').fetchall()
        random.shuffle(rows)
        rows = rows[:100]

    def set_mode(mode: str):
        if mode not in {'train', 'validate', 'test'}:
            raise ValueError('invalid dataset mode')

        self.mode = mode


dl = DataLoader(TickExamples(mode='train'),
                batch_size=1,
                shuffle=False)
