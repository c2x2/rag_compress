from baseline.langchain import LangchainRag
from baseline.llama import LlamaRag
from data_progress.base import TriviaqaDataload
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    dataloader = TriviaqaDataload("/data/datasets/.json", "triviaqa_web","web")
    data_path = "/data/datasets/triviaq_web.json"
    with open(data_path, 'r') as f:
        datasets = json.load(f)
    langchainrag = LangchainRag("langchain")
    langchainrag.load_data(datasets[:16], "triviaqa_web")
    langchainrag.run()
