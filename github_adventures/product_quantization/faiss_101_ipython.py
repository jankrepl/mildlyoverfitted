import faiss
import numpy as np

# Load fast text embeddings
embs = np.load("parsed_fasttext/embs.npy")  # change path if necessary
embs.shape
embs.nbytes / 1e6

# Prepare parameters
d = embs.shape[1]
m = 10
nbits = 8
k = 2 ** nbits
k

# Construct index
index = faiss.IndexPQ(d, m, nbits)
index.is_trained

# Try encoding without any training
index.sa_encode(embs[:2])

# Train the model
index.train(embs)
index.is_trained
index.ntotal

# Add vectors to the database
index.add(embs)
index.ntotal

codes = faiss.vector_to_array(index.codes).reshape(index.ntotal, m)
codes[:3]
codes.nbytes / 1e6

# Try searching - EXHAUSTIVE SEARCH
index.search(embs[:3], 4)

# Quickly show that with flat index distances are precise
flat_index = faiss.IndexFlatL2(d)
flat_index.train(embs)
flat_index.add(embs)
flat_index.search(embs[:3], 4)
