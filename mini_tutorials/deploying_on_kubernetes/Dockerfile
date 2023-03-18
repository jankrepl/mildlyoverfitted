FROM huggingface/transformers-pytorch-gpu

RUN python3 -c "from transformers import AutoModel;AutoModel.from_pretrained('bert-base-uncased')"
RUN python3 -c "from transformers import AutoTokenizer;AutoTokenizer.from_pretrained('bert-base-uncased')"

RUN pip install fastapi uvicorn

EXPOSE 8888
ENTRYPOINT ["transformers-cli", "serve", "--port=8888", "--host=0.0.0.0", "--task=fill-mask", "--model=bert-base-uncased"]
