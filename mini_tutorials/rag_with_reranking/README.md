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
