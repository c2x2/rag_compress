from baseline.rag_base import BaseRag
from data_progress.base import QAItem
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
from typing import List, Optional, TypedDict
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from llama_index.llms.dashscope import DashScope
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from ragas import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall
)
from datasets import Dataset
from transformers import AutoTokenizer
load_dotenv()

class Ragas(TypedDict):
    question:List[str]
    answer:List[str]
    contexts:List[List[str]]
    ground_truth:List[str]

class Compress(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List["NodeWithScore"], query_bundle: Optional[QueryBundle] = None
    ) -> List["NodeWithScore"]:

        for node in nodes:
            node.node.text = node.node.text[:50]
        
        return nodes

class LlamaRag(BaseRag):
    llm:DashScope
    embedding:HuggingFaceEmbedding
    tokenizer:AutoTokenizer
    use_compress:bool
    def __init__(self, name: str, use_compress:bool = False):
        super().__init__(name)
        self.llm = DashScope(
            model=str(os.getenv("LLM_NAME")),  # qwen-plus
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0
        )

        self.embedding = HuggingFaceEmbedding(
            model_name=str(os.getenv("EMBEDDING_PATH")),
            device="cuda"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(os.getenv("TOKENIZER_PATH")),
            trust_remote_code=True
        )

        self.token_counter = TokenCountingHandler(
            tokenizer=lambda x: self.tokenizer.encode(x)
        )

        self.callback_manager = CallbackManager([self.token_counter])
        self.use_compress = use_compress
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
        self.evalute_llm = HuggingFacePipeline(pipeline=pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, max_length=None))
        self.evalute_embedding = HuggingFaceEmbeddings(model_name=str(os.getenv("EMBEDDING_PATH")), model_kwargs={"device": "cuda"})

        #设置模型
        Settings.llm = self.llm
        Settings.embed_model = self.embedding
        Settings.callback_manager = self.callback_manager

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
                embeddings=self.evalute_embedding
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
                embeddings=self.evalute_embedding
            )
            result = origin_results.to_pandas().to_dict()

        return result
        
    def compress(self, docs):
         return super().compress(docs)
    
    def run(self):
        """
        进行测试
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

        for qa in tqdm(self.datasets[:1], desc="start query"):
            documents = [Document(text=content) for content in qa["contents"]]
            print(documents)
            #construct index vector
            index = VectorStoreIndex.from_documents(documents)
            answer = qa["answer"]
            query = qa["query"]

            if not self.use_compress:
                #caculate origin
                self.token_counter.reset_counts()
                origin_engine = index.as_query_engine(similarity_top_k=10)
                response = origin_engine.query(query)
                origin_answer["answer"].append(response.__str__())
                origin_answer["question"].append(query)
                origin_answer["ground_truth"].append(answer)

                contexts = []
                for node in response.source_nodes:
                    contexts.append(node.node.text)

                origin_answer["contexts"].append(contexts)

                origin_tokens["total_tokens"].append(self.token_counter.total_llm_token_count)
                origin_tokens["prompt_tokens"].append(self.token_counter.prompt_llm_token_count)
                origin_tokens["completion_tokens"].append(self.token_counter.completion_llm_token_count)
            
            else:
                #caculate origin
                self.token_counter.reset_counts()
                compress_engin = index.as_query_engine(similarity_top_k=10, node_postprocessor=[Compress])
                response = compress_engin.aquery(query)
                compressed_answer["answer"].append(response.__str__())
                compressed_answer["question"].append(query)
                compressed_answer["ground_truth"].append(answer)

                contexts = []
                for node in response.source_nodes:
                    contexts.append(node.node.text)

                compressed_answer["contexts"].append(contexts)

                compressed_tokens["total_tokens"].append(self.token_counter.total_llm_token_count)
                compressed_tokens["prompt_tokens"].append(self.token_counter.prompt_llm_token_count)
                compressed_tokens["completion_tokens"].append(self.token_counter.completion_llm_token_count)
        
        compressed_result = {
            "tokens": compressed_tokens,
            "answer":compressed_answer
        }
        origin_result = {
            "tokens": origin_tokens,
            "answer":origin_answer
        }
        
        # result = self.measure(origin_result, compressed_result)
        result = origin_result if not self.use_compress else compressed_result
        
        with open(f"/root/works/rag_compress/results/{self.name}/{self.__get_compress()}_{self.dataname}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"结果已保存至./result/{self.name}/{self.__get_compress()}_{self.dataname}.json")
        
        return
