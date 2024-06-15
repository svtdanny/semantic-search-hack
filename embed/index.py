import pickle
from sklearn.neighbors import NearestNeighbors

from settings import settings

class KnnIndex:
    def __init__(self):
        with open(settings.precalculated_embeddings, "rb") as f:
            self.joined_result = pickle.load(f)

        self.urls = []
        self.embeddings = []
        for link, vector in self.joined_result.items():
            self.add_to_storage(link, vector)

        self.rebuild_index()

    def rebuild_index(self):
        self.neigh = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.neigh.fit(self.embeddings)

    def add_to_storage(self, link, vector):
        self.urls.append(link)
        self.embeddings.append(vector)

    def search(self, query_embedding, n_items=10):
        indxs = self.neigh.kneighbors(query_embedding, n_items, return_distance=False).ravel().tolist()
        return [self.urls[indx] for indx in indxs]
