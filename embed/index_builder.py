from knn_index import KnnIndex
from qdrant_index import QDrantIndex

from settings import settings

def build_index():
    if settings.use_qdrant_index:
        return QDrantIndex() # fast but external dependency
    else:
        return KnnIndex() # slow but generic
