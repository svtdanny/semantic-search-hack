from tqdm import tqdm
import pickle
from settings import settings
from itertools import islice

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

class QDrantIndex:
    def __init__(self, embeddding_dim=1024, location=":memory:", collection_name = "search_index"):
        self.upload_chunk_preload = 10000

        self.embeddding_dim = embeddding_dim
        self.location = location
        self.collection_name = collection_name
        self.init_client()

        with open(settings.precalculated_embeddings, "rb") as f:
            self.joined_result = pickle.load(f)

        self.preload_storage()

    def init_client(self):
        self.client  = QdrantClient(self.location)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embeddding_dim, distance=Distance.DOT),
        )

        self.inserted = 0 # in production use uuid4 for id (no collision statistically) or store in persistent storage

    def preload_storage(self):
        total=int( (len(self.joined_result)+self.upload_chunk_preload-1)/self.upload_chunk_preload )
        for batch in tqdm(batched(self.joined_result, self.upload_chunk_preload), total=total):
            points = [PointStruct(id=self.inserted+i, vector=self.joined_result[batch[i]], payload={"link": batch[i]}) for i in range(len(batch))]
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
            )
            self.inserted += len(points)

    def add_to_storage(self, link, vector):
        points = [PointStruct(id=self.inserted, vector=vector, payload={"link": link})]
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )
        self.inserted += 1

    def search(self, query_embedding, n_items=10):
        search_result = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=10
        )
        return [res.payload["link"] for res in search_result]
