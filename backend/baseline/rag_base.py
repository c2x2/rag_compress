from backend.data_progress.base import QAItem
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
    
    def compress(self, docs, query):
        raise NotImplementedError

    def run_demo(self, contexts, query, use_compress):
        raise NotImplementedError
