# Description
## Installation

Run the following command to deploy a simple OpenSearch DB locally.
 
```bash
docker run -p 9200:9200 -p 9600:9600 -e "DISABLE_SECURITY_PLUGIN=true" -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:latest
```
The version of the image was `2.10.0` at the time of making the video.

To install the Python dependencies run
```bash
pip install opensearch-py cohere
```
Again, I did not hardcode any version, but the versions at the time of
making the video were

```bash
cohere==4.27
opensearch-py==2.3.1
```

## Contents
* `answer.py` - scripts that does RAG question answering - requires question as the only argument
* `input.txt` - each line corresponds to a document to be added to OpenSearch(except for emtpy lines and comments)
* `upload_data.py` - load `input.txt` into OpenSearch


Note that to use the `answer.py` you need to get a Cohere API token and
then export 
```bash
export COHERE_API_KEY=VERYSECRET
python answer.py 'What is the meaning of life?'
```

## Postman
You can import the `postman_collection.json` in Postman and then
simply add the following 3 variables in your environment

* `OpenSearchURL` - will be `http://localhost:9200` if you follow the above instructions
* `CohereURL` - should be `https://api.cohere.ai/v1`
* `CohereAPIKey` - you need to generate this yourself

# Diagrams

## RAG with embeddings
<img width="1165" alt="rag-with-embeddings" src="https://github.com/jankrepl/mildlyoverfitted/assets/18519371/678e69eb-96a9-4fa1-bcff-8c848d69f10a">

## RAG with reranking
<img width="1169" alt="rag-with-reranking" src="https://github.com/jankrepl/mildlyoverfitted/assets/18519371/45ea091b-5724-4117-bfec-d219afdd9f40">
