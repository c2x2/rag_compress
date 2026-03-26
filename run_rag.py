from backend.baseline.langchain import LangchainRag
from backend.baseline.llama import LlamaRag
from backend.baseline.light import LightRag
from backend.data_progress.base import TriviaqaDataload
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    # dataloader = TriviaqaDataload("/home/melonmelon/agent/rag_compress/datasets/triviaqa/qa/web-dev.json", "triviaqa_web","web")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "eval"], default="eval")    
    data_path = "./backend/data/triviaq_web.json"
    with open(data_path, 'r') as f:
        datasets = json.load(f)
    # datasets = dataloader.load()
    rag = LlamaRag()
    print(len(datasets))
    rag.load_data(datasets[:8], "triviaqa_web")

    args = parser.parse_args()

    if args.mode == "run":
        rag.run()
    else:
        rag.measure(None, None)

