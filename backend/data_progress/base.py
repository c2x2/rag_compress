import json
from typing import List, Dict, TypedDict, TypeVar, Generic
from dotenv import load_dotenv
import os
from tqdm import tqdm
load_dotenv()
class QAItem(TypedDict):
    query:str
    answer:str
    contents:List[str]
T = TypeVar("T")

class database(Generic[T]):
    """
    最终输出格式如下的数据:
    [
    {"query":query,"answer":answer,"contents":["","",""...]},...
    ]
    """
    def __init__(self, path : str, dataname:str):
        print(f"load data from {path}")
        self.path = path
        self.dataname = dataname
        return 
    def __get_name(self):
        return self.dataname

    def load(self)-> List[T]:
        raise NotImplementedError 
        
class TriviaqaDataload(database[QAItem]):
    def __init__(self, path, dataname:str, data_name:str) -> None:
        super().__init__(path, dataname)
        self.data_name=data_name

    def load(self)-> List[QAItem]:

        file_size = os.path.getsize(self.path)

        with open(self.path, 'r', encoding='utf-8') as f:
            content = ""
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading JSON") as pbar:
                for line in f:
                    content += line
                    pbar.update(len(line))

        raw_data = json.loads(content)
        data_results:List[QAItem] = []
        #TODO fix numbers
        for qa in tqdm(raw_data['Data'][:10000], desc="Load datasets"):
            envidences_path = []
            item: QAItem = {
                "answer":qa["Answer"]["Value"],
                "query":qa["Question"],
                "contents":[]
            }
            
            if self.data_name=="wiki":
                envidences_path=[source["Filename"] for source in qa["EntityPages"]]
            elif self.data_name=="web":
                envidences_path=[source["Filename"] for source in qa["SearchResults"]]
            
            for file_path in envidences_path:
                # with open(f"{os.getenv('ROOT_PATH')}/datasets/triviaqa/evidence/{self.data_name}/{file_path}","r", encoding='utf-8') as f:
                with open(f"/home/melonmelon/agent/rag_compress/datasets/triviaqa/evidence/{self.data_name}/{file_path}","r", encoding='utf-8') as f:
                    content = f.read()
                item["contents"].append(content)

            data_results.append(item)
        
        return data_results
    

# if __name__ == "__main__":
#     test = TriviaqaDataload("/home/melonmelon/agent/rag_compress/datasets/triviaqa/qa/web-dev.json", "web")
#     print(test.load()[:1])