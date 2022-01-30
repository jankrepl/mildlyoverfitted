# GPT-2 custom implementation
## Installation

```python
pip install -r requirements.txt
```

## Launching script
To copy weights of an official model + generate some text use the script
`copy_and_generate.py`

```python
(gpt) gpt$ python copy_and_generate.py --help
usage: Copy weights of a HF model and generate text. [-h] [--sample] [-s STEPS] [-r RANDOM_STATE]
                                                     [-t TEMPERATURE] [-k TOP_K] [-v]
                                                     {gpt2,gpt2-medium,gpt2-large,distilgpt2}
                                                     initial_text

positional arguments:
  {gpt2,gpt2-medium,gpt2-large,distilgpt2}
                        Pretrained model to use
  initial_text          Initial text

optional arguments:
  -h, --help            show this help message and exit
  --sample              If True sample randomly otherwise take the most probable token (default: False)
  -s STEPS, --steps STEPS
                        Number of new tokens to generate (default: 30)
  -r RANDOM_STATE, --random-state RANDOM_STATE
                        Random state (default: None)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Softmax logits temperature (default: 1)
  -k TOP_K, --top-k TOP_K
                        If specified, then selecting k most probable tokens (default: None)
  -v, --verbose         If True, then verbose (default: False)

```
