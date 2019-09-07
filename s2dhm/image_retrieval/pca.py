import numpy as np
from sklearn.decomposition import PCA


def normalize(array, axis=-1):
    """Normalize array."""
    return np.array(array) / np.linalg.norm(array, axis=axis, keepdims=True)

def learn_and_apply_pca(ref_desc, qry_desc, ndim=1024):
    """Learn and apply PCA."""
    pca = PCA(n_components=min(ndim, len(ref_desc)))
    # Learn PCA on reference descriptors
    ref_desc = normalize(pca.fit_transform(normalize(ref_desc)))
    # Apply it on the query descriptors
    qry_desc = normalize(pca.transform(normalize(qry_desc)))
    return ref_desc, qry_desc
