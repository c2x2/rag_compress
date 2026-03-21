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
        # self.llm = ChatOpenAI(model=str(os.getenv("LLM_NAME")), api_key=str(os.getenv("LLM_API_KEY")), base_url=str(os.getenv("LLM_BASE_URL")))
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
        self.evalute_llm = HuggingFacePipeline(pipeline=pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, batch_size=8, max_length=None))
        self.llm = self.evalute_llm
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(os.getenv("EMBEDDING_PATH")), 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={
                "batch_size": 32,   # 🔥 核心优化
                "normalize_embeddings": True
            }
        )
        self.prompt_template = """
        Given these texts:
        -----
        {context}
        -----
        Please answer the following question:
        {query}
        """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "query"])
        self.use_compress = use_compress
        self.batch_size = 16
        # self.retriver = None

    def __get_compress(self) -> str:
        return "use_compress" if self.use_compress else "no_compress"

    def load_data(self, datasets: List[QAItem], dataname:str):
        self.datasets = datasets
        self.dataname = dataname

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
    def run(self):
        """
        对每个qa进行问答测试
        """
        origin_answer:Ragas={
                    "question":[],
                    "answer":[],
                    "contexts":[],
                    "ground_truth":[]
                }
        origin_tokens = {
            "total_tokens":[],
            "prompt_tokens":[],
            "completion_tokens":[]
                }
        compressed_answer:Ragas={
                    "question":[],
                    "answer":[],
                    "contexts":[],
                    "ground_truth":[]
                }
        compressed_tokens = {
            "total_tokens":[],
            "prompt_tokens":[],
            "completion_tokens":[]
        }
        chain = create_stuff_documents_chain(self.llm, self.prompt)

        for qa in tqdm(self.datasets, desc="start query"):
            #创建retriver
            retriever = InMemoryVectorStore.from_texts(qa["contents"], self.embedding).as_retriever(search_kwargs={"k": 10})
            query = qa['query']
            answer = qa["answer"]
            origin_docs = retriever.invoke(query)
            #记录原始内容
            origin_answer['question'].append(query)
            origin_answer["contexts"].append([doc.page_content for doc in origin_docs])
            origin_answer['ground_truth'].append(answer)

            #记录压缩内容
            compressed_docs = self.compress(origin_docs)
            compressed_answer["question"].append(query)
            compressed_answer["contexts"].append([doc.page_content for doc in compressed_docs])
            compressed_answer["ground_truth"].append(answer)
            
            if not self.use_compress:
                # 计算原始token消耗
                with get_openai_callback() as origin_cb:
                    chain = create_stuff_documents_chain(self.llm, self.prompt)
                    response = chain.invoke({"context":origin_docs, "query":query})
                    origin_answer['answer'].append(response)
                    origin_tokens["completion_tokens"].append(origin_cb.completion_tokens)
                    origin_tokens['prompt_tokens'].append(origin_cb.prompt_tokens)
                    origin_tokens["total_tokens"].append(origin_cb.total_tokens)
            else:
                # 计算压缩后token消耗
                with get_openai_callback() as compressed_cb:
                    chain = create_stuff_documents_chain(self.llm, self.prompt)
                    response = chain.invoke({"context":compressed_docs, "query":query})
                    compressed_answer['answer'].append(response)
                    compressed_tokens["completion_tokens"].append(compressed_cb.completion_tokens)
                    compressed_tokens['prompt_tokens'].append(compressed_cb.prompt_tokens)
                    compressed_tokens["total_tokens"].append(compressed_cb.total_tokens)
        
        compressed_result = {
            "tokens": compressed_tokens,
            "answer":compressed_answer
        }
        origin_result = {
            "tokens": origin_tokens,
            "answer":origin_answer
        }
        #计算结果
        # result = self.measure(origin_result, compressed_result)
        result = origin_result if self.use_compress else compressed_result
        filepath = f"/root/works/rag_compress/results/{self.name}/{self.__get_compress()}_{self.dataname}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"结果已保存至{filepath}")
        return
