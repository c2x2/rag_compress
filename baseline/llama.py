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
from llama_index.core.callbacks import CallbackManager
from ragas import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

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
    llm:AutoModelForCausalLM
    embedding:HuggingFaceEmbedding
    use_compress:bool
    def __init__(self, name: str, use_compress:bool = False):
        super().__init__(name)
        self.batch_size = 4
        self.top_k = 5
        self.max_context_len = 1200
        self.max_new_tokens = 256
        #定义模型
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")
        self.model = model
        model.config.use_cache = False
        #评测用模型
        self.evalute_llm = HuggingFacePipeline(pipeline=pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, batch_size=self.batch_size, max_length=None))
        self.embedding = HuggingFaceEmbedding(
            model_name=str(os.getenv("EMBEDDING_PATH")), 
            device="cuda",
            embed_batch_size=self.batch_size,
            normalize=True
        )
        self.use_compress = use_compress  
        self.mycompress = Compress()

        #设置模型
        Settings.embed_model = self.embedding

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
        
    def compress(self, docs):
         return super().compress(docs)
    
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
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(text)

        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
            return_tensors="pt"
        ).to("cuda")

        # 记录输入张量的宽度（这是所有 prompt 补齐后的统一长度）
        input_tensor_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        responses = []
        prompt_tokens_counts = []
        completion_tokens_counts = []
        total_tokens_counts = []

        for i in range(len(outputs)):
            # 1. 获取该行真正的 prompt 长度（不含 padding）
            actual_prompt_len = inputs["attention_mask"][i].sum().item()
            
            # 2. 正确切片：从 input_tensor_len 之后开始才是真正的生成内容
            # 无论前面有多少 PAD，生成的回答始终跟在整个输入张量后面
            generated_tokens = outputs[i][input_tensor_len:]
            
            # 3. 过滤掉生成结果末尾可能的填充 token (如果有的话)
            # 寻找第一个 EOS token 之后的位置并截断，或者直接用 skip_special_tokens
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 4. 计算真实的 Token 数量
            # 生成的有效 token 数需要排除掉生成阶段产生的 [PAD]
            # 我们通过解码后再编码，或者直接统计非 pad 的 token
            actual_gen_tokens = (generated_tokens != self.tokenizer.pad_token_id).sum().item()

            responses.append(text)
            prompt_tokens_counts.append(actual_prompt_len)
            completion_tokens_counts.append(actual_gen_tokens)
            total_tokens_counts.append(actual_prompt_len + actual_gen_tokens)

        return responses, prompt_tokens_counts, completion_tokens_counts, total_tokens_counts
    
    def run(self):
        """
        进行测试
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
                documents = [Document(text=content) for content in qa["contents"]]
                #construct index vector
                index = VectorStoreIndex.from_documents(documents)
                if self.use_compress:
                    engine = index.as_retriever(similarity_top_k = self.top_k, node_postprocessors=[self.mycompress])
                else:
                    engine = index.as_retriever(similarity=self.top_k)
                
                docs = [data.text for data in engine.retrieve(QueryBundle(qa["query"]))]

                context_text = '\n'.join([d for d in docs])
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
                target["contexts"].append([d for d in docs])
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
        root_path = str(os.getenv("ROOT_PATH"))
        filepath = f"{root_path}/results/{self.name}/{self.__get_compress()}_{self.dataname}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"结果已保存至{filepath}")
        return