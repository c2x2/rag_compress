from data_progress.base import QAItem
from typing import List

class BaseRag:
    def __init__(self, name:str):
        self.name = name
    
    def load_data(self, datasets : List[QAItem], dataname:str):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError

    def measure(self, origin, compressed):
        raise NotImplementedError
    
    def compress(self, docs):
        raise NotImplementedError

