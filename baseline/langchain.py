from baseline.rag_base import BaseRag
from dotenv import load_dotenv
import os
import json
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.callbacks import get_openai_callback
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from typing import Sequence, List, TypedDict
from data_progress.base import QAItem
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall
)
from datasets import Dataset
import numpy as np
import torch
from tqdm import tqdm
load_dotenv()

class Ragas(TypedDict):
    question:List[str]
    answer:List[str]
    contexts:List[List[str]]
    ground_truth:List[str]

class LangchainRag(BaseRag):
    llm:HuggingFacePipeline
    embedding:HuggingFaceEmbeddings
    prompt_template:str
    prompt:PromptTemplate
    use_compress:bool
    def __init__(self, name:str, use_compress:bool=False):
        super().__init__(name)
        self.batch_size = 2
        self.top_k = 5
        self.max_context_len = 1200
        self.max_new_tokens = 256
        # self.llm = ChatOpenAI(model=str(os.getenv("LLM_NAME")), api_key=str(os.getenv("LLM_API_KEY")), base_url=str(os.getenv("LLM_BASE_URL")))
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")
        self.model = model
        model.config.use_cache = False 
        self.evalute_llm = HuggingFacePipeline(pipeline=pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, batch_size=self.batch_size, max_length=None))
        self.llm = self.evalute_llm
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(os.getenv("EMBEDDING_PATH")), 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={
                "batch_size": self.batch_size,   # 🔥 核心优化
                "normalize_embeddings": True
            }
        )
        self.use_compress = use_compress
        # self.retriver = None

    def __get_compress(self) -> str:
        return "use_compress" if self.use_compress else "no_compress"

    def load_data(self, datasets: List[QAItem], dataname:str):
        self.datasets = datasets
        self.dataname = dataname

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def measure(self, origin, compressed):
        
        if self.use_compress:
            #计算压缩后answer质量
            dataset = Dataset.from_dict(compressed['answer'])
            compressed_results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    answer_correctness,
                    context_recall
                ],
                llm=self.evalute_llm,
                embeddings=self.embedding
            )
            result = compressed_results.to_pandas().to_dict()
        else:
            #计算压缩前answer质量
            dataset = Dataset.from_dict(origin['answer'])
            origin_results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    answer_correctness,
                    context_recall
                ],
                llm=self.evalute_llm,
                embeddings=self.embedding
            )
            result = origin_results.to_pandas().to_dict()

        return result
        #计算压缩后和压缩前比较tokens降低程度
        # origin_tokens = np.array(origin["tokens"]["total_tokens"])
        # compressed_tokens = np.array(compressed["tokens"]["total_tokens"])

        # compression_ratio = (origin_tokens - compressed_tokens) / np.maximum(origin_tokens, 1)
        # mean_sample_radio = np.mean(compression_ratio)
        # mean_dataset_radio = np.mean(compression_ratio)
        
        # return{
        #     "compressed_results":compressed_results.to_pandas().to_dict(),
        #     "origin_results":origin_results.to_pandas().to_dict(),
        #     "mean_sample":float(mean_sample_radio),
        #     "mean_dataset":float(mean_dataset_radio)
        # }


    def compress(self, docs:List[Document]) -> List[Document]:
        compressed_docs = []
        for doc in docs:
            compressed_docs.append(Document(doc.page_content[:50]))
        
        return compressed_docs
    
    def build_prompt(self, context, query):
        return f"""
                Given these texts:
                -----
                {context}
                -----
                Please answer the following question:
                {query}
                """
    def generate_batch(self, prompts):

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
            return_tensors="pt"
        ).to("cuda")

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=False   # 🔥关键
            )

        responses = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        prompt_tokens = [len(ids) for ids in input_ids]
        total_tokens = [len(out) for out in outputs]
        completion_tokens = [
            t - p for t, p in zip(total_tokens, prompt_tokens)
        ]

        return responses, prompt_tokens, completion_tokens, total_tokens

    def run(self):
        """
        对每个qa进行问答测试
        """
        target: Ragas = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        token_stats = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }

        for i in tqdm(range(0, len(self.datasets), self.batch_size), desc="start query"):
            batch = self.datasets[i:i+self.batch_size]

            batch_prompts = []
            batch_docs = []
            batch_qas = []

            for qa in batch:
                retriever = InMemoryVectorStore.from_texts(
                    qa["contents"], self.embedding
                ).as_retriever(search_kwargs={"k": self.top_k})

                docs = retriever.invoke(qa["query"])

                if self.use_compress:
                    docs = self.compress(docs)

                # 🔥 控制context长度
                context_text = "\n".join([d.page_content for d in docs])
                context_text = context_text[:self.max_context_len]

                prompt = self.build_prompt(context_text, qa["query"])

                batch_prompts.append(prompt)

                batch_docs.append(docs)
                batch_qas.append(qa)

            responses, ptks, ctks, ttks = self.generate_batch(batch_prompts)

            # 写入
            for qa, docs, resp, pt, ct, tt in zip(
                batch_qas, batch_docs, responses, ptks, ctks, ttks
            ):

                target["question"].append(qa["query"])
                target["contexts"].append([d.page_content for d in docs])
                target["ground_truth"].append(qa["answer"])
                target["answer"].append(resp)

                token_stats["prompt_tokens"].append(int(pt))
                token_stats["completion_tokens"].append(int(ct))
                token_stats["total_tokens"].append(int(tt))

            torch.cuda.empty_cache()

        final_output = {
            "data": target,
            "tokens": token_stats
        }
        filepath = f"/root/works/rag_compress/results/{self.name}/{self.__get_compress()}_{self.dataname}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"结果已保存至{filepath}")
        return
