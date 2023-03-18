# Relevant commands

## Creating an API
```bash
transformers-cli serve --task=fill-mask --model=bert-base-uncased
```

```bash
curl http://localhost:8888 | jq 
```

```bash
curl -X POST http://localhost:8888/forward  -H "accept: application/json" -H "Content-Type: application/json" -d '{"inputs": "Today is going to be a [MASK] day"}' | jq 
```

## Containerization
Build first image.
```bash
docker build -t cool-api:v1 .
```

Build second image.
```bash
docker build -t cool-api:v2 -f DockerfileConda .
```

Run image.
```bash
docker run -it --rm -P cool-api:v2
```

## Deploying on Kubernetes
Start a minikube cluster.
```bash
minikube start
```

Get all objects across all namespaces.
```bash
kubectl get all -A
```

List images.
```bash
minikube image list
```

Load an image.
```bash
minikube image cool-api:v2
```

Create a deployment.
```bash
kubectl create deploy cool-deploy --image=cool-api:v2
```

Create a service.
```bash
kubectl expose deploy/cool-deploy --name=cool-service --target-port=8888 --port=1234
```

Scale up.
```bash
kubectl scale deploy/cool-deploy --replicas=3
```

Get logs.
```bash
kubectl logs -f PODFULLNAME
```
