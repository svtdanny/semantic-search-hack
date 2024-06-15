from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np

from embed.embedder import IndexEmbeddingModel
from embed.index import KnnIndex

index = KnnIndex()
embedder = IndexEmbeddingModel()
app = FastAPI()


class SearchItem(BaseModel):
    links: List[str]

class SearchRequest(BaseModel):
    query: str

@app.put("/search")
async def put_new_key(item: SearchItem):
    for link in item.links:
        index.add_to_storage(link, embedder.get_video_embedding(link))

@app.post("/search")
async def get_search_results(request: SearchRequest):
    query_embedding = embedder.get_query_embedding(request.query)
    res = index.search(query_embedding)
    return res
