"""
dedupe.py
----------
Deduplication strategies:
1. Embedding similarity (ArcFace via InsightFace, FAISS if available).
2. Perceptual hash (pHash) fallback.
"""

import numpy as np
from PIL import Image
import imagehash

HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def phash_dedupe(img_paths):
    """Deduplicate by perceptual hash."""
    keep, seen = [], set()
    for idx, path in enumerate(img_paths):
        try:
            h = str(imagehash.phash(Image.open(path).convert("RGB")))
            if h not in seen:
                seen.add(h)
                keep.append(idx)
        except Exception:
            keep.append(idx)
    return keep


def dedupe_embeddings(feats, cosine_thresh=0.92):
    """
    Deduplicate normalized embeddings by cosine similarity.
    feats: numpy array (N x D).
    """
    n = feats.shape[0]
    keep_mask = np.ones(n, dtype=bool)

    if HAS_FAISS:
        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats)
        D, I = index.search(feats, 10)
        for i in range(n):
            if not keep_mask[i]:
                continue
            for sim, j in zip(D[i][1:], I[i][1:]):  # skip self
                if sim >= cosine_thresh and keep_mask[j]:
                    keep_mask[j] = False
    else:
        for i in range(n):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n):
                if keep_mask[j] and np.dot(feats[i], feats[j]) >= cosine_thresh:
                    keep_mask[j] = False

    return np.where(keep_mask)[0]
