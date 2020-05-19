from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from bbo_simulator import TickMessage, BboSimulator
from ib_common import *


class SimpleStrategy:

    def __init__(self):

        mask = {TICK_DELAYED_LAST, }

        self.bbo_service = BboSimulator(mask)

        self.msg_queue = Queue()

    def run(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.bbo_service.subscribe(self.msg_queue))

        while not self.msg_queue.empty():
            print(self.msg_queue.qsize())

            tick = self.msg_queue.get()

            print(tick.tick_type)


simple_strategy = SimpleStrategy()
simple_strategy.run()
