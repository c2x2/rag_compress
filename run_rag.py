from backend.baseline.langchain import LangchainRag
from backend.baseline.llama import LlamaRag
from backend.baseline.light import LightRag
from backend.data_progress.base import TriviaqaDataload
from backend.evals.eval import EvalEngine
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "eval"], default="eval")    
    args = parser.parse_args()

    if args.mode == "run":
        dataloader = TriviaqaDataload("/home/melonmelon/agent/rag_compress/datasets/triviaqa/qa/web-train.json", "triviaqa_web","web")
        data_path = "backend/baseline/datasets/triviaq_web.json"
        with open(data_path, 'w') as f:
            json.dump(dataloader.load(), f)
        exit(0)
        datasets = dataloader.load()
        rag = LlamaRag()
        print(len(datasets))
        rag.load_data(datasets, "triviaqa_web")

        rag.run()
    else:

        data_path = '/home/melonmelon/agent/server/rag_compress/backend/results/langchain/no_compress_triviaqa_web.json'
        
        engine = EvalEngine("langchain")
        engine.load_data(data_path)
        engine.eval()