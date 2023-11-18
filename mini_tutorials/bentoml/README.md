1. [Resources](#resources)
2. [Installation](#installation)
3. [Instructions](#instructions)
    1. [`bentoml`](#bentoml)
    1. [`bentoctl`](#bentoml)
4. [Sketches](#sketches)

# Resources
* https://docs.bentoml.com/en/latest/
* https://github.com/bentoml/bentoctl
* https://github.com/bentoml/aws-sagemaker-deploy

# Installation
```bash
pip install -r requirements.txt
```

# Instructions
## `bentoml`
Creating a model
```bash
python create_model.py
```

Listing all existing models
```bash
bentoml models list
```

Build a bento
```bash
bentoml build
```

List all existing bentos
```bash
bentoml list
```

Serve a bento locally
```bash
bentoml serve $BENTO
```

Serve a `service.py` (development)
```bash
bentoml serve service.py
```

## `bentoctl`

## `aws` CLI


# Sketches

