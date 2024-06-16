from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np
import time

from embed.embedder import IndexEmbeddingModel
from embed.index_builder import build_index

index = build_index()
embedder = IndexEmbeddingModel()
app = FastAPI()


class SearchItem(BaseModel):
    links: List[str]

class SearchRequest(BaseModel):
    query: str

@app.put("/add")
async def put_new_key(item: SearchItem):
    for link in item.links:
        index.add_to_storage(link, embedder.get_video_embedding(link))

@app.post("/query")
async def get_search_results(request: SearchRequest):
    start = time.time()
    query_embedding = embedder.get_query_embedding(request.query)
    print("Query embed time: ", time.time() - start)

    start = time.time()
    res = index.search(query_embedding[0])
    print("Query search time: ", time.time() - start)
    return res
