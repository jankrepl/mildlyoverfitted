# Installation

Run the following to get all the dependencies.

```
pip install -r requirements.txt
```

# Faiss 101

The code for the short intro to FAISS can be found in `faiss_101_ipython.py`.
Note that you can use `parse.py` to turn the raw fasttext embeddings
into a numpy array. See `run_all.sh` for example usage.

# Custom PQ implementation

The custom PQ implementation can be found inside of `custom.py`.

# End to end script

The script `run_all.sh` does the following things:

* Download fasttext embeddings
* Train multiple indexes (faiss + custom) using the embeddings
* Serve gradio apps for similarity search comparing different indexes

```
chmod +x run_all.sh
./run_all
```

Don't forget to kill the Gradio processes by `pkill -f gradio` once you
don't need them anymore.
