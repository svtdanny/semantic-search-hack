from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np

from embed.controller import KnnIndex

index = KnnIndex()
app = FastAPI()


class SearchItem(BaseModel):
    links: List[str]

class SearchRequest(BaseModel):
    query: str

@app.put("/search")
async def put_new_key(item: SearchItem):
    print(f"Put {item.links}")
    for link in item.links:
        index.add_to_storage(link, np.zeros((1024,))) # stup vector

@app.post("/search")
async def get_search_results(request: SearchRequest):
    res = index.search(request.query)
    return res
