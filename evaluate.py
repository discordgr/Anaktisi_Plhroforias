from search_engineV2 import (
    load_inverted_index,
    load_documents,
    preprocess_documents,
    initialize_vsm,
    initialize_bm25,
    process_query
)

# 2. Ορίζεις μια συνάρτηση evaluate_system
def evaluate_system(test_queries, inverted_index, vectorizer, tfidf_matrix, bm25, documents):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_queries = len(test_queries)

    for test_query in test_queries:
        query = test_query["query"]
        relevant_docs = set(test_query["relevant_docs"])

        # Παίρνεις τα αποτελέσματα από το process_query
        ranked_docs = process_query(
            query=query,
            inverted_index=inverted_index,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            bm25=bm25,
            documents=documents,
            default_operator="or"
        )

        retrieved_docs = [doc_id for (doc_id, _) in ranked_docs]

        # Υπολογισμός precision, recall, f1
        retrieved_set = set(retrieved_docs)
        true_positives = len(retrieved_set & relevant_docs)

        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        print(f"Query: {query}")
        print(f"  Relevant Docs: {relevant_docs}")
        print(f"  Retrieved Docs (top 5): {retrieved_docs[:5]}")
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}\n")

    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = total_f1 / num_queries

    print("=== AVERAGE RESULTS ===")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall:    {avg_recall:.2f}")
    print(f"Average F1:        {avg_f1:.2f}")


def main():
    # Αρχεία
    index_file = "inverted_index_tfidf.json"
    docs_file = "wiki_definitions_cleaned.json"

    inverted_index = load_inverted_index(index_file)
    if not inverted_index:
        print("Could not load inverted index.")
        return

    documents = load_documents(docs_file)
    if not documents:
        print("Could not load documents.")
        return

    # Προετοιμασία για VSM & BM25
    print("\nPreprocessing documents for VSM and BM25...")
    processed_texts = preprocess_documents(documents)
    print("Initializing Vector Space Model (VSM)...")
    vectorizer, tfidf_matrix = initialize_vsm(processed_texts)
    print("Initializing Okapi BM25...")
    bm25 = initialize_bm25(processed_texts)

    test_queries = [
        {
            "query": "machine learning",
            "relevant_docs": [
                3, 4, 10, 11, 13, 16, 17, 18, 19,
                21, 22, 40, 41, 42, 65, 23, 45, 46, 53,
                58, 64, 74, 82, 84, 85, 86,
                88, 89, 90, 92, 93, 94, 95, 96, 97, 98,
                99, 100
            ]
        },
        {
            "query": "neural network",
            "relevant_docs": [
                4, 7, 8, 9, 10, 19, 23, 24, 25, 28, 46,
                48, 83, 85, 88, 89, 90, 91
            ]
        },
        {
            "query": "data science",
            "relevant_docs": [
                3, 4, 5, 6, 8, 9, 10, 11, 13, 14,
                16, 17, 19, 22, 24, 25, 31, 34,
                35, 38, 39, 41, 42, 44, 50, 51, 53,
                54, 55, 56, 57, 59, 68, 70, 73, 78,
                82, 83, 91, 92, 97
            ]
        }
    ]


    print("\n=== START EVALUATION ===")
    evaluate_system(test_queries, inverted_index, vectorizer, tfidf_matrix, bm25, documents)

if __name__ == "__main__":
    main()
