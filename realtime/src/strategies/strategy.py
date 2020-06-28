from abc import ABC, abstractmethod


class Strategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def tick(self,
             timestamp: int,
             price: float,
             size: int):
        pass
