import os
import sys


import cohere
from opensearchpy import OpenSearch

# Helper
def generate_prompt(question: str, contexts: str):
    prompt = (
        "Given the following extracted parts of a long document and a "
        'question, create a final answer with references ("SOURCES").'
        "If you don't know the answer, just say that you don't know, don't try "
        'to make up an answer. ALWAYS return a "SOURCES" part in your answer.\n'
    )

    prompt += f"QUESTION: {question}\n"
    prompt += "".join(
        [f"SOURCE {i}: {context}\n" for i, context in enumerate(contexts)]
    )
    prompt += "ANSWER: "

    return prompt


# PARAMETERS
INDEX_NAME = "cool_index"
FIELD_NAME = "stuff"
RETRIEVER_K = 5
RERANKER_K = 2
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

question = sys.argv[1]

# Instantiate clients
os_client = OpenSearch(
    hosts=[
        {
            "host": "localhost",
            "port": 9200,
        }
    ]
)
cohere_client = cohere.Client(COHERE_API_KEY)

# Retrieve
os_results = os_client.search(
    body={
        "query": {
            "match": {
                FIELD_NAME: question
            }
        }
    },
    size=RETRIEVER_K
)
contexts = [x["_source"][FIELD_NAME] for x in os_results["hits"]["hits"]]
print("OpenSearch: ", contexts)

# Rerank
cohere_results = cohere_client.rerank(
    model="rerank-english-v2.0",
    query=question,
    documents=contexts,
    top_n=RERANKER_K,
)
reranked_contexts = [r.document["text"] for r in cohere_results]
print("Cohere Reranked: ", reranked_contexts)


# Chat completion
prompt = generate_prompt(question, reranked_contexts)

response = cohere_client.chat(
    chat_history=[],
    message=prompt
)

print("Answer: ", response.text)
