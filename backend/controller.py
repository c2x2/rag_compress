from backend.baseline.langchain import LangchainRag
from backend.baseline.llama import LlamaRag
from backend.baseline.light import LightRag
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import torch
load_dotenv()
class RAGController:
    def __init__(self, texts) -> None:
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")

        self.systems={
            "LangChain":LangchainRag(tokenizer=tokenizer, model=model),
            "LlamaIndex":LlamaRag(tokenizer=tokenizer, model=model),
            # "light":LightRag()
        }
        self.texts = texts

    def run(self, system, query, compress):
        return self.systems[system].run_demo(self.texts, query, compress)