from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle
from typing import List, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

# ===============================
# 1️⃣ tokenizer（用于统计token）
# ===============================

tokenizer = AutoTokenizer.from_pretrained(
    str(os.getenv("TOKENIZER_PATH")),
    trust_remote_code=True
)
token_counter = TokenCountingHandler(
    tokenizer=lambda x: tokenizer.encode(x)
)

callback_manager = CallbackManager([token_counter])

Settings.callback_manager = callback_manager


# ===============================
# 2️⃣ embedding
# ===============================

Settings.embed_model = HuggingFaceEmbedding(
    model_name=str(os.getenv("EMBEDDING_PATH")),
    device="cuda"
)

# ===============================
# 3️⃣ LLM
# ===============================

Settings.llm = DashScope(
    model=str(os.getenv("LLM_NAME")),  # qwen-plus
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0
)


# ===============================
# 4️⃣ 自定义 reranker
# ===============================

class MyCustomReranker(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List["NodeWithScore"], query_bundle: Optional[QueryBundle] = None
    ) -> List["NodeWithScore"]:

        print(f"原始召回数量: {len(nodes)}")

        # reranked_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)[:1]

        for node in nodes:
            node.node.text = node.node.text[:5]

        return nodes


# ===============================
# 5️⃣ 构建 index
# ===============================

documents = [
    Document(text="多智能体系统可以解决复杂规划问题..."),
    Document(text="北京今天天气很好..."),
    Document(text="我吃了中午饭..."),
    Document(text="我要写作业..."),
    Document(text="我我的我 啊我...")
]

index = VectorStoreIndex.from_documents(documents)


# ==========================================
# 对照组：Naive RAG
# ==========================================

token_counter.reset_counts()

naive_engine = index.as_query_engine(similarity_top_k=5)
result = naive_engine.retrieve("多智能体系统有什么用？")
print(result)


my_reranker = MyCustomReranker()
experiment_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[my_reranker]
)
result = experiment_engine.retrieve(QueryBundle("多智能体系统有什么用？"))
print([doc.text for doc in result])

exit(0)
# response_naive = naive_engine.query("多智能体系统有什么用？")

# print("Naive回答：", response_naive)
# # 拿到检索到的节点
# source_nodes = response_naive.source_nodes

# for i, node in enumerate(source_nodes):
#     print(f"\n--- Context {i} ---")
#     print(node.node.text)
#     print("score:", node.score)

# print("\nNaive Token统计")
# print("Prompt tokens:", token_counter.prompt_llm_token_count)
# print("Completion tokens:", token_counter.completion_llm_token_count)
# print("Total tokens:", token_counter.total_llm_token_count)


# ==========================================
# 实验组：加入你的 Reranker
# ==========================================

# token_counter.reset_counts()

# my_reranker = MyCustomReranker()

# experiment_engine = index.as_query_engine(
#     similarity_top_k=5,
#     node_postprocessors=[my_reranker]
# )

# response_experiment = experiment_engine.query("多智能体系统有什么用？")

# print("Experiment回答：", response_experiment)
# source_nodes = response_experiment.source_nodes

# for i, node in enumerate(source_nodes):
#     print(f"\n--- Context {i} ---")
#     print(node.node.text)
#     print("score:", node.score)
# print("\nExperiment Token统计")
# print("Prompt tokens:", token_counter.prompt_llm_token_count)
# print("Completion tokens:", token_counter.completion_llm_token_count)
# print("Total tokens:", token_counter.total_llm_token_count)