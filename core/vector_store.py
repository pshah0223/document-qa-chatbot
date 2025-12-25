import faiss
import pickle
import numpy as np
from typing import List, Dict


class FaissStore:
    def __init__(self, dim:int):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []
    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        self.index.add(vectors.astype(np.float32))
        self.meta.extend(metadata)
    def search(self, q_vec: np.ndarray, top_k=10):
        scores, idxs = self.index.search(q_vec.astype(np.float32), top_k)
        out = []
        for i in range(top_k):
            idx = idxs[0][i]
            if idx == -1:
                continue
            out.append((float(scores[0][i]), self.meta[idx]))
        return sorted(out, key=lambda x: -x[0])
    def save(self, index_path, meta_path):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f)
    def load(self, index_path, meta_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
