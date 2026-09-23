import math
import re


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


def tokenize(text):
    words = re.findall(
        r"\b[a-zA-Z0-9\-]+\b",
        text.lower()
    )

    return set(words)


def keyword_overlap(question, chunk):
    question_words = tokenize(question)
    chunk_words = tokenize(chunk)

    if not question_words:
        return 0.0

    overlap = question_words.intersection(
        chunk_words
    )

    return len(overlap) / len(question_words)


def retrieve_top_chunks(
    client,
    question,
    chunks,
    embeddings,
    top_k=6,
    model="text-embedding-3-small",
    semantic_weight=0.8,
    keyword_weight=0.2
):
    question_response = client.embeddings.create(
        model=model,
        input=question
    )

    question_embedding = (
        question_response
        .data[0]
        .embedding
    )

    scores = []

    for i, chunk_embedding in enumerate(embeddings):

        semantic_score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        keyword_score = keyword_overlap(
            question,
            chunks[i]
        )

        final_score = (
            semantic_weight * semantic_score
            +
            keyword_weight * keyword_score
        )

        scores.append(
            (
                final_score,
                semantic_score,
                keyword_score,
                i
            )
        )

    scores.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return scores[:top_k]