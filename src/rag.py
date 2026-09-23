import os
import json
import math

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# -----------------------------
# 1. Load API key
# -----------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# -----------------------------
# 2. File paths
# -----------------------------

pdf_path = (
    "data/"
    "JIMSS2025_Modeling_of_abrasive_"
    "magnetorheological_fluids_for_"
    "drag_finishing_revision2.pdf"
)

cache_path = "data/embeddings.json"


# -----------------------------
# 3. Split text into chunks
# -----------------------------

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# 4. Cosine similarity
# -----------------------------

def cosine_similarity(a, b):
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    return dot_product / (norm_a * norm_b)


# -----------------------------
# 5. Load or create embeddings
# -----------------------------

if os.path.exists(cache_path):

    print("Loading cached embeddings...")

    with open(
        cache_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    chunks = data["chunks"]

    page_numbers = data["page_numbers"]

    embeddings = data["embeddings"]


else:

    print("Creating embeddings...")

    reader = PdfReader(pdf_path)

    chunks = []

    page_numbers = []

    for page_index, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            page_chunks = split_text(text)

            for chunk in page_chunks:

                chunks.append(chunk)

                page_numbers.append(
                    page_index + 1
                )


    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = [
        item.embedding
        for item in embedding_response.data
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


# -----------------------------
# 6. Ask user question
# -----------------------------

question = input(
    "\nAsk a question about the paper: "
)


# -----------------------------
# 7. Embed the question
# -----------------------------

question_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

question_embedding = (
    question_response
    .data[0]
    .embedding
)


# -----------------------------
# 8. Compare similarity
# -----------------------------

scores = []

for i, chunk_embedding in enumerate(embeddings):

    score = cosine_similarity(
        question_embedding,
        chunk_embedding
    )

    scores.append(
        (score, i)
    )


scores.sort(
    reverse=True
)


# -----------------------------
# 9. Get top 3 chunks
# -----------------------------

top_results = scores[:3]


# -----------------------------
# 10. Build context
# -----------------------------

context_parts = []

for score, index in top_results:

    page = page_numbers[index]

    text = chunks[index]

    context_parts.append(
        f"[Page {page}]\n{text}"
    )


context = "\n\n".join(
    context_parts
)


# -----------------------------
# 11. Build RAG prompt
# -----------------------------

prompt = f"""
You are an engineering research assistant.

Answer the question using only the retrieved paper context below.

When possible, cite the relevant page in the form [Page X].

Do not invent information that is not contained in the retrieved context.

If the retrieved context is insufficient, say clearly:
"The retrieved paper context does not contain enough information."

Retrieved context:

{context}

Question:

{question}
"""


# -----------------------------
# 12. Generate answer
# -----------------------------

answer_response = client.responses.create(
    model="gpt-5.6-luna",
    input=prompt
)


# -----------------------------
# 13. Print answer
# -----------------------------

print("\nRAG answer:\n")

print(
    answer_response.output_text
)


# -----------------------------
# 14. Print sources
# -----------------------------

print("\nSources:")

for score, index in top_results:

    print(
        f"- Page {page_numbers[index]} "
        f"| Chunk {index} "
        f"| similarity = {score:.4f}"
    )