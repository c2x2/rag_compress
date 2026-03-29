from backend.baseline.langchain import LangchainRag
from backend.baseline.llama import LlamaRag
from backend.baseline.light import LightRag
from backend.data_progress.base import TriviaqaDataload
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "eval"], default="eval")    
    args = parser.parse_args()

    if args.mode == "run":
        dataloader = TriviaqaDataload("/home/melonmelon/agent/rag_compress/datasets/triviaqa/qa/verified-web-dev.json", "triviaqa_web","web")
        # data_path = "backend/baseline/datasets/triviaq_web.json"
        # with open(data_path, 'r') as f:
        #     # json.dump(dataloader.load(), f)
        #     datasets = json.load(f)
        # exit(0)
        datasets = dataloader.load()
        rag = LangchainRag(use_compress=True)
        rag.load_data(datasets[:8], "triviaqa_web")

        rag.run()
    else:
        from backend.evals.eval import EvalEngine

        data_path = '/home/melonmelon/agent/server/rag_compress/backend/results/langchain/no_compress_triviaqa_web.json'
        
        engine = EvalEngine("langchain")
        engine.load_data(data_path)
        engine.eval()