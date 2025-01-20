import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

def load_inverted_index(index_file):
    """
    Φορτώνει το inverted index από ένα JSON αρχείο.
    """
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file {index_file}.")
        return None

def load_documents(docs_file):
    """
    Φορτώνει τη λίστα των εγγράφων (με paragraphs) από JSON αρχείο.
    """
    try:
        with open(docs_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file {docs_file}.")
        return None

def preprocess_text(text):
    """
    Προεπεξεργάζεται το κείμενο.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def preprocess_documents(documents):
    """
    Προεπεξεργάζεται όλα τα έγγραφα.
    """
    processed_texts = []
    for doc in documents:
        original_def = doc.get('original_definition', '')
        tokens = preprocess_text(original_def)
        processed_text = ' '.join(tokens)
        processed_texts.append(processed_text)
    return processed_texts

def initialize_vsm(processed_texts):
    """
    Αρχικοποιεί το Vector Space Model (VSM).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, tfidf_matrix

def initialize_bm25(processed_texts):
    """
    Αρχικοποιεί το Okapi BM25.
    """
    tokenized_texts = [text.split() for text in processed_texts]
    bm25 = BM25Okapi(tokenized_texts)
    return bm25

def process_boolean_query(tokens, inverted_index):
    """
    Επεξεργάζεται ένα ερώτημα Boolean.
    """
    result_set = None
    operation = None

    for token in tokens:
        if token in ["and", "or", "not"]:
            operation = token
        else:
            docs_for_token = {int(doc) for doc in inverted_index.get(token, [])}

            if result_set is None:
                result_set = docs_for_token
            else:
                if operation == "and":
                    result_set &= docs_for_token
                elif operation == "or":
                    result_set |= docs_for_token
                elif operation == "not":
                    result_set -= docs_for_token
                else:
                    result_set |= docs_for_token

    return result_set if result_set is not None else set()

def process_simple_query(tokens, inverted_index, default_operator="or"):
    """
    Επεξεργάζεται απλά queries χωρίς Boolean operators.
    """
    result_set = set()
    for i, token in enumerate(tokens):
        docs_for_token = {int(doc) for doc in inverted_index.get(token, [])}
        if i == 0:
            result_set = docs_for_token
        else:
            if default_operator == "or":
                result_set |= docs_for_token
            elif default_operator == "and":
                result_set &= docs_for_token
    return result_set

def calculate_ranking_boolean(result_docs, inverted_index, query_terms, vectorizer, tfidf_matrix):
    """
    Υπολογίζει το TF-IDF score για Boolean queries.
    """
    doc_scores = {}
    for term in query_terms:
        if term in vectorizer.vocabulary_:
            term_index = vectorizer.vocabulary_[term]
            term_tfidf = tfidf_matrix[:, term_index].toarray().flatten()
            for doc_id in result_docs:
                doc_id = int(doc_id)
                score = term_tfidf[doc_id - 1]
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
    return doc_scores

def calculate_ranking_vsm(query, vectorizer, tfidf_matrix):
    """
    Υπολογίζει τα cosine similarity scores για VSM.
    """
    query_vector = vectorizer.transform([query])
    return cosine_similarity(query_vector, tfidf_matrix).flatten()

def calculate_ranking_bm25(query, bm25):
    """
    Υπολογίζει τα BM25 scores.
    """
    tokens = query.split()
    return bm25.get_scores(tokens)

def process_query(query, inverted_index, vectorizer, tfidf_matrix, bm25, documents, default_operator="or"):
    """
    Ενιαία συνάρτηση για Boolean και simple search.
    """
    tokens = query.lower().split()
    has_boolean_ops = any(tok in ["and", "or", "not"] for tok in tokens)

    if has_boolean_ops:
        stemmed_tokens = preprocess_text(query)
        result_docs = process_boolean_query(stemmed_tokens, inverted_index)
        if result_docs:
            query_terms = [tok for tok in stemmed_tokens if tok not in ["and", "or", "not"]]
            doc_scores = calculate_ranking_boolean(result_docs, inverted_index, query_terms, vectorizer, tfidf_matrix)
            ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked_docs = []
    else:
        stemmed_tokens = preprocess_text(query)
        result_docs = process_simple_query(stemmed_tokens, inverted_index, default_operator)
        if result_docs:
            print("\nSelect Ranking Algorithm:")
            print("1. TF-IDF Sum")
            print("2. Vector Space Model (Cosine Similarity)")
            print("3. Okapi BM25")
            choice = input("Enter the number of the ranking algorithm (1-3): ")

            if choice == "1":
                query_terms = [tok for tok in stemmed_tokens]
                doc_scores = calculate_ranking_boolean(result_docs, inverted_index, query_terms, vectorizer, tfidf_matrix)
                ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            elif choice == "2":
                original_query = ' '.join(stemmed_tokens)
                cosine_similarities = calculate_ranking_vsm(original_query, vectorizer, tfidf_matrix)
                ranked_docs = sorted([(doc_id, cosine_similarities[doc_id - 1]) for doc_id in result_docs], key=lambda x: x[1], reverse=True)
            elif choice == "3":
                original_query = ' '.join(stemmed_tokens)
                bm25_scores = calculate_ranking_bm25(original_query, bm25)
                ranked_docs = sorted([(doc_id, bm25_scores[doc_id - 1]) for doc_id in result_docs], key=lambda x: x[1], reverse=True)
            else:
                print("Invalid choice. Defaulting to TF-IDF Sum.")
                query_terms = [tok for tok in stemmed_tokens]
                doc_scores = calculate_ranking_boolean(result_docs, inverted_index, query_terms, vectorizer, tfidf_matrix)
                ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked_docs = []

    return ranked_docs

def main():
    index_file = "inverted_index_tfidf.json"
    docs_file = "wiki_definitions_cleaned.json"

    inverted_index = load_inverted_index(index_file)
    if not inverted_index:
        return

    documents = load_documents(docs_file)
    if not documents:
        return

    print("Preprocessing documents for VSM and BM25...")
    processed_texts = preprocess_documents(documents)

    print("Initializing Vector Space Model (VSM)...")
    vectorizer, tfidf_matrix = initialize_vsm(processed_texts)

    print("Initializing Okapi BM25...")
    bm25 = initialize_bm25(processed_texts)

    print("\nWelcome to the Enhanced Search Engine!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        ranked_docs = process_query(query, inverted_index, vectorizer, tfidf_matrix, bm25, documents, default_operator="or")

        if ranked_docs:
            print(f"\nFound {len(ranked_docs)} documents:")
            for doc_id, score in ranked_docs:
                real_index = doc_id - 1
                if 0 <= real_index < len(documents):
                    doc = documents[real_index]
                    title = doc.get('title', 'No Title')
                    original_def = doc.get('original_definition', 'No Definition')
                    snippet = original_def[:200] + "..." if len(original_def) > 200 else original_def
                    print(f"  Document ID: {doc_id}")
                    print(f"    Title: {title}")
                    print(f"    Definition: {snippet}")
                    print(f"    Score: {score:.4f}\n")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
