import os
import json

from openai import OpenAI


def load_or_create_embeddings(
    client: OpenAI,
    chunks,
    page_numbers,
    cache_path="data/embeddings.json",
    model="text-embedding-3-small"
):
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")

        with open(cache_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        embeddings = data["embeddings"]
        cached_chunks = data["chunks"]
        cached_page_numbers = data["page_numbers"]

        return (
            cached_chunks,
            cached_page_numbers,
            embeddings
        )

    print("Creating embeddings...")

    response = client.embeddings.create(
        model=model,
        input=chunks
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    data = {
        "chunks": chunks,
        "page_numbers": page_numbers,
        "embeddings": embeddings
    }

    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file
        )

    print("Embeddings saved.")

    return (
        chunks,
        page_numbers,
        embeddings
    )