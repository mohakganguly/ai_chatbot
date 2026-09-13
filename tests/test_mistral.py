from graph.routing import QueryRouter


router = QueryRouter()

queries = [
    "Hello, how are you?",
    "What is machine learning?",
]

for i, query in enumerate(queries, start=1):
    print(f"\n--- Router Request {i} ---")
    print("Query:", query)

    route = router.route(query)

    print("Route:", route)