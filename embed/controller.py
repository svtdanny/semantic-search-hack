import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

import video_clip

DEVICE = "cpu"

class KnnIndex:
    def __init__(self):
        with open("embed/joined_embeddings.pkl", "rb") as f:
            self.joined_result = pickle.load(f)

        self.urls = []
        self.embeddings = []
        for link, vector in self.joined_result.items():
            self.add_to_storage(link, vector)
            # self.urls.append(link)
            # self.embeddings.append(vector)

        self.rebuild_index()

        eval_config = 'embed/eval_configs/video_clip_v0.2.yaml'
        self.model, vis_processor = video_clip.load_model(eval_config, DEVICE)
        self.model = self.model.to(DEVICE)
        self.model = self.model.eval()


    def rebuild_index(self):
        self.neigh = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.neigh.fit(self.embeddings)

    def add_to_storage(self, link, vector):
        self.urls.append(link)
        self.embeddings.append(vector)

    def search(self, query, n_items=10):
        query_embedding = video_clip.embed_text_itc(self.model, query, DEVICE).cpu().numpy()

        indxs = self.neigh.kneighbors(query_embedding, n_items, return_distance=False).ravel().tolist()
        return [self.urls[indx] for indx in indxs]

if __name__ == "__main__":
    index = KnnIndex()
    res = index.search("minecraft")
    print(res)