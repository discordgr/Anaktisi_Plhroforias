import json
from collections import defaultdict
import math

def create_inverted_index_with_tfidf(input_json, output_json):

    # 1. Φορτώνουμε το καθαρισμένο JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_docs = len(data)  # Πόσα έγγραφα έχουμε συνολικά

    # 2. Δομή για να αποθηκεύουμε τη συχνότητα (TF) των λέξεων:
    freq_index = defaultdict(lambda: defaultdict(int))

    # 3. Για κάθε έγγραφο, αυξάνουμε τη συχνότητα για κάθε λέξη
    for i, doc in enumerate(data):
        doc_id = i + 1  
        tokens = doc.get('cleaned_definition', [])

        for token in tokens:
            freq_index[token][doc_id] += 1

    # 4. Υπολογίζουμε το IDF για κάθε λέξη:
    # Όπου df = πόσα έγγραφα περιέχουν τη λέξη
    idf_scores = {}
    for term, docs_dict in freq_index.items():
        df = len(docs_dict)  # πόσα έγγραφα έχουν αυτή τη λέξη
        idf_scores[term] = math.log(total_docs / (1 + df))

    inverted_index = {}

    for term, docs_dict in freq_index.items():
        inverted_index[term] = {}
        for doc_id, tf in docs_dict.items():
            tf_idf = tf * idf_scores[term] 
            inverted_index[term][doc_id] = {
                "tf": tf,
                "tf-idf": tf_idf
            }

    # 6. Αποθηκεύουμε σε αρχείο JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)

    print(f"Inverted index with TF and TF-IDF saved to {output_json}")

if __name__ == "__main__":
    input_file = "wiki_definitions_cleaned.json"
    output_file = "inverted_index_tfidf.json"

    create_inverted_index_with_tfidf(input_file, output_file)
