from pathlib import Path
from opensearchpy import OpenSearch

INPUT_FILE = "input.txt"
INDEX_NAME = "cool_index"
FIELD_NAME = "stuff"

client = OpenSearch(
    hosts=[
        {
            "host": "localhost",
            "port": 9200,
        }
    ]
)

print(client.ping())

with Path(INPUT_FILE).open() as f:
    i = 0
    for line in f.read().splitlines():
        if not line or line.startswith("#"):
            continue

        print(f"Adding {i}")
        client.index(index=INDEX_NAME, body={FIELD_NAME: line})
        i += 1
