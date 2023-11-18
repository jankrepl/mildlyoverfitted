1. [Resources](#resources)
2. [Installation](#installation)
3. [Instructions](#instructions)
    1. [`bentoml`](#bentoml)
    1. [`bentoctl`](#bentoctl)
    1. [`aws` CLI](#aws-cli)
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
Install SageMaker operator
```bash
bentoctl operator install aws-sagemaker
```

Initialize
```bash
bentoctl init
```

ATTENTION: All of the below assumes that you have correctly set up AWS
secret keys and permissions.

Build custom customized SageMaker image and push to ECR

```bash
bentoctl build -f deployment_config.yaml -b $BENTO
```


Initialize terraform
```bash
terraform init
```

Look at what changes will be applied

```bash
terraform plan -var-file=bentoctl.tfvars
```

Actually apply changes
```bash
terraform apply -var-file=bentoctl.tfvars
```

Send request to the API Gateway
```bash
curl -X 'POST' "$URL/classify"   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{            
  "sepal_width": 0,
  "sepal_length": 0,
  "petal_width": 0,
  "petal_length": 0
}'
```

Destroy resources (not including ECR)

```bash
terraform destroy -var-file=bentoctl.tfvars
```

Destroy resources  including ECR)

```bash
bentoctl destroy
```

## `aws` CLI
Describe repositories
```bash
aws ecr describe-repositories
```

List all images in the repository `amazing-iris`
```bash
aws ecr list-images --repository-name=amazing-iris
```

List SageMaker models
```bash
aws sagemaker list-models
```

List SageMaker endpoints
```bash
aws sagemaker list-endpoints
```


# Sketches
<img width="1250" alt="bentoml-overview" src="https://github.com/jankrepl/mildlyoverfitted/assets/18519371/eeb60c7f-2bbd-40df-9d89-95c0a720c16b">
<img width="1069" alt="sklearn-sagemaker" src="https://github.com/jankrepl/mildlyoverfitted/assets/18519371/58848152-cffb-4ec2-8b8f-b25a3d6647c0">
