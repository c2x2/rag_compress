from fastapi import FastAPI
from backend.controller import RAGController

app = FastAPI()

# demo数据
texts = open("backend/data/demo.txt").read().split("\n")

controller = RAGController(texts)

@app.post("/query")
def query_api(data: dict):

    # print("test")
    # return "test"
    result = controller.run(
        system=data["system"],
        query=data["query"],
        compress=data["compress"]
    )
    return result