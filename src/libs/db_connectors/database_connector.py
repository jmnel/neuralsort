from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Type
from pathlib import Path


class DatabaseConnector(ABC):

    @staticmethod
    @abstractmethod
    def connect(*args, **kwargs) -> Type[DatabaseConnector]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_tables(self) -> List[Tuple[Union[int, str, float], ...]]:
        pass

    @abstractmethod
    def table_exists(self, name: str) -> bool:
        pass

    @abstractmethod
    def get_schema(self, table_name) -> List[Dict[str, Union[str, int, bool]]]:
        pass

    @abstractmethod
    def insert(self,
               table: str,
               w_columns: List[str],
               values: List[Tuple[Union[str, int, float], ...]]):
        pass

    @abstractmethod
    def select(self,
               table: str,
               w_columns: List[str],
               w_filter: str):
        pass

    @abstractmethod
    def delete(self, table: str, w_filter: str):
        pass

    @abstractmethod
    def create_table(self, table_name: str,
                     columns: List[Dict[str, Union[int, float, str, bool]]]):
        pass

    @abstractmethod
    def drop_table(self, table: str):
        pass

    @abstractmethod
    def commit(self):
        pass
