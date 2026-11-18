import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    similarity = dot_product / (norm_a * norm_b)

    return similarity

from faqs import faqs

questions = [faq["question"] for faq in faqs]

faq_embeddings = model.encode(questions)

print("FAQ Embeddings Created:", faq_embeddings.shape)

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])[0]

    scores = []

    for i, faq_embedding in enumerate(faq_embeddings):
        score = cosine_similarity(query_embedding, faq_embedding)

        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []

    for index, score in scores[:top_k]:
        results.append({
            "question": faqs[index]["question"],
            "answer": faqs[index]["answer"],
            "category": faqs[index]["category"],
            "score": float(score)
        })

    return results

def keyword_search(query):
    results = []

    query_words = query.lower().split()

    for faq in faqs:
        question_words = faq["question"].lower().split()
        answer_words = faq["answer"].lower().split()

        all_words = question_words + answer_words

        if any(word in all_words for word in query_words):
            results.append(faq)

    return results

