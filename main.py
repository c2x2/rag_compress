from baseline.langchain import LangchainRag
from baseline.llama import LlamaRag
from data_progress.base import TriviaqaDataload
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    # dataloader = TriviaqaDataload("/data/datasets/.json", "triviaqa_web","web")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "eval"], default="eval")    
    data_path = "./data/triviaq_web.json"
    with open(data_path, 'r') as f:
        datasets = json.load(f)
    langchainrag = LangchainRag("langchain")
    langchainrag.load_data(datasets[:8], "triviaqa_web")

    args = parser.parse_args()

    if args.mode == "run":
        langchainrag.run()
    else:
        langchainrag.measure(None, None)

