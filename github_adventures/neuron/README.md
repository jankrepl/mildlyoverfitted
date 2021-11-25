# Installation

```bash
pip install -r requirements.txt
```

# Running training
To run the same experiments as in the video run

```bash
./launch.sh
```

However, feel free to check the contents of the `launch.sh` for single
experiments.

# Evaluation and pretrained models
This repo contains multiple pretrained models inside of `pretrained/`. They
are all `.pkl` files and they were created by pickling `solutions.Solution`
subclasses. To load them inside of Python run something along these lines

```python
import pickle

solution_path = "pretrained/invariant_ours.pkl"  # you can change this

with open(solution_path, "rb") as f:
    solution = pickle.load(f)[0]

```

You can also run any of the below scripts to reproduce the results from
the end of the video.


```bash
EPISODES=30

python evaluate_shuffling.py -e $EPISODES
python evaluate_noise.py -e $EPISODES
python evaluate_video.py -e $EPISODES
```
