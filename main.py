from search import semantic_search, keyword_search

queries = [
    "I cannot log in",
    "when is salary credited",
    "work from home rules"
]

for query in queries:
    print("\nQuery:", query)

    print("\nKeyword Search:")
    print(keyword_search(query))

    print("\nSemantic Search:")
    print(semantic_search(query, top_k=3))

    print("-" * 50)