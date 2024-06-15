import io
from urllib.request import urlopen

from settings import settings
import video_clip

def video2embedding(video: io.BytesIO, model, vls_processor, device, reduction="mean"): 
    embedding = video_clip.get_all_video_embeddings([video], model, vls_processor, device)
    if reduction == "mean":
        embedding = embedding[0].mean(dim=1).ravel()
    return embedding

class IndexEmbeddingModel:
    def __init__(self):
        eval_config = settings.eval_config
        self.model, self.vis_processor = video_clip.load_model(eval_config, settings.device)
        self.model = self.model.to(settings.device)
        self.model = self.model.eval()

    def get_video_embedding(self, link):
        try:
            rsp = urlopen(link)
        except Exception as e:
            print("Some problem with video link: ", e)
            raise

        embedding = video2embedding(io.BytesIO(rsp.read()), self.model, self.vis_processor, settings.device)
        return embedding

    def get_query_embedding(self, query):
        embedding = video_clip.embed_text_itc(self.model, query, settings.device).cpu().numpy()
        return embedding
