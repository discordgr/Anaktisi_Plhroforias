import requests
from bs4 import BeautifulSoup
import json

def fetch_html(url):
    """
    Στέλνει αίτημα GET σε ένα URL και επιστρέφει το HTML της σελίδας.
    Επιστρέφει None αν υπάρξει σφάλμα δικτύου ή άλλο HTTP σφάλμα.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_first_paragraph(html):
    """
    Αναλύει το HTML με BeautifulSoup και επιστρέφει τον τίτλο και την πρώτη
    παράγραφο.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Αφαιρούμε tags <sup>
    for sup_tag in soup.find_all('sup'):
        sup_tag.decompose()

    # Αφαιρούμε tags <span class="reference">
    for ref_tag in soup.find_all('span', class_='reference'):
        ref_tag.decompose()

    # Βρίσκουμε τον τίτλο της σελίδας
    title_tag = soup.find('title')
    title = title_tag.text if title_tag else "No Title"

    # Βρίσκουμε όλες τις παραγράφους <p>
    paragraphs = soup.find_all('p')

    # Παίρνουμε την πρώτη μη κενή παράγραφο
    first_paragraph = ""
    for p in paragraphs:
        # Χρησιμοποιούμε separator=" " για να υπάρχει κενό μεταξύ text nodes
        text = p.get_text(separator=" ", strip=True)
        if text:
            first_paragraph = text
            break

    return title, first_paragraph

def collect_wikipedia_definitions(url_list, output_filename="wiki_definitions.json"):
    """
    Δέχεται μια λίστα από Wikipedia URLs
    """
    collected_data = []

    for url in url_list:
        print(f"Fetching: {url}")
        html = fetch_html(url)
        if html:
            title, first_paragraph = extract_first_paragraph(html)
            data = {
                'url': url,
                'title': title,
                'definition': first_paragraph
            }
            collected_data.append(data)
        else:
            # Αν αποτυγχάνει το fetch, καταγράφουμε το σφάλμα
            data = {
                'url': url,
                'title': "Error",
                'definition': "Could not fetch page"
            }
            collected_data.append(data)

    # Αποθήκευση σε αρχείο JSON
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=4)

    print(f"\nTotal pages fetched: {len(collected_data)}")
    print(f"Data saved to '{output_filename}'")


if __name__ == "__main__":
    wikipedia_pages = [
        "https://en.wikipedia.org/wiki/Web_crawler",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Data_science",
        "https://en.wikipedia.org/wiki/Big_data",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Internet_of_things",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Computer_vision",
        "https://en.wikipedia.org/wiki/Robotics",
        "https://en.wikipedia.org/wiki/Data_mining",
        "https://en.wikipedia.org/wiki/Information_retrieval",
        "https://en.wikipedia.org/wiki/Cloud_computing",
        "https://en.wikipedia.org/wiki/Supervised_learning",
        "https://en.wikipedia.org/wiki/Unsupervised_learning",
        "https://en.wikipedia.org/wiki/Reinforcement_learning",
        "https://en.wikipedia.org/wiki/Support-vector_machine",
        "https://en.wikipedia.org/wiki/Decision_tree",
        "https://en.wikipedia.org/wiki/Random_forest",
        "https://en.wikipedia.org/wiki/Gradient_boosting",
        "https://en.wikipedia.org/wiki/Computer_cluster",
        "https://en.wikipedia.org/wiki/Parallel_computing",
        "https://en.wikipedia.org/wiki/Distributed_computing",
        "https://en.wikipedia.org/wiki/Concurrent_computing",
        "https://en.wikipedia.org/wiki/High-performance_computing",
        "https://en.wikipedia.org/wiki/Bayesian_network",
        "https://en.wikipedia.org/wiki/Genetic_algorithm",
        "https://en.wikipedia.org/wiki/Speech_recognition",
        "https://en.wikipedia.org/wiki/Automatic_summarization",
        "https://en.wikipedia.org/wiki/Information_extraction",
        "https://en.wikipedia.org/wiki/Recommender_system",
        "https://en.wikipedia.org/wiki/Dimensionality_reduction",
        "https://en.wikipedia.org/wiki/Principal_component_analysis",
        "https://en.wikipedia.org/wiki/Clustering",
        "https://en.wikipedia.org/wiki/K-means_clustering",
        "https://en.wikipedia.org/wiki/DBSCAN",
        "https://en.wikipedia.org/wiki/Hierarchical_clustering",
        "https://en.wikipedia.org/wiki/Association_rule_learning",
        "https://en.wikipedia.org/wiki/Knowledge_discovery_in_databases",
        "https://en.wikipedia.org/wiki/Data_warehouse",
        "https://en.wikipedia.org/wiki/ETL_(Extract,_transform,_load)",
        "https://en.wikipedia.org/wiki/Data_visualization",
        "https://en.wikipedia.org/wiki/Matplotlib",
        "https://en.wikipedia.org/wiki/TensorFlow",
        "https://en.wikipedia.org/wiki/PyTorch",
        "https://en.wikipedia.org/wiki/Keras",
        "https://en.wikipedia.org/wiki/Scikit-learn",
        "https://en.wikipedia.org/wiki/Apache_Spark",
        "https://en.wikipedia.org/wiki/Apache_Hadoop",
        "https://en.wikipedia.org/wiki/MongoDB",
        "https://en.wikipedia.org/wiki/PostgreSQL",
        "https://en.wikipedia.org/wiki/Relational_database",
        "https://en.wikipedia.org/wiki/NoSQL",
        "https://en.wikipedia.org/wiki/Database_index",
        "https://en.wikipedia.org/wiki/SQL",
        "https://en.wikipedia.org/wiki/Machine_reading_comprehension",
        "https://en.wikipedia.org/wiki/Text_mining",
        "https://en.wikipedia.org/wiki/Question_answering",
        "https://en.wikipedia.org/wiki/Chatbot",
        "https://en.wikipedia.org/wiki/GPT-3",
        "https://en.wikipedia.org/wiki/BERT_(language_model)",
        "https://en.wikipedia.org/wiki/Word_embedding",
        "https://en.wikipedia.org/wiki/Vector_space_model",
        "https://en.wikipedia.org/wiki/Boolean_retrieval",
        "https://en.wikipedia.org/wiki/Okapi_BM25",
        "https://en.wikipedia.org/wiki/Lucene",
        "https://en.wikipedia.org/wiki/ElasticSearch",
        "https://en.wikipedia.org/wiki/Apache_Solr",
        "https://en.wikipedia.org/wiki/Inverse_document_frequency",
        "https://en.wikipedia.org/wiki/Tf–idf",
        "https://en.wikipedia.org/wiki/Information_retrieval#Evaluation",
        "https://en.wikipedia.org/wiki/Precision_and_recall",
        "https://en.wikipedia.org/wiki/F1_score",
        "https://en.wikipedia.org/wiki/Mean_average_precision",
        "https://en.wikipedia.org/wiki/Text_classification",
        "https://en.wikipedia.org/wiki/Document_classification",
        "https://en.wikipedia.org/wiki/Topic_modeling",
        "https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation",
        "https://en.wikipedia.org/wiki/Word2vec",
        "https://en.wikipedia.org/wiki/Convolutional_neural_network",
        "https://en.wikipedia.org/wiki/Recurrent_neural_network",
        "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
        "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
        "https://en.wikipedia.org/wiki/Long_short-term_memory",
        "https://en.wikipedia.org/wiki/Gated_recurrent_unit",
        "https://en.wikipedia.org/wiki/Generative_adversarial_network",
        "https://en.wikipedia.org/wiki/Autoencoder",
        "https://en.wikipedia.org/wiki/Multilayer_perceptron",
        "https://en.wikipedia.org/wiki/Feedforward_neural_network",
        "https://en.wikipedia.org/wiki/Pattern_recognition",
        "https://en.wikipedia.org/wiki/Semi-supervised_learning",
        "https://en.wikipedia.org/wiki/Active_learning_(machine_learning)",
        "https://en.wikipedia.org/wiki/Transfer_learning",
        "https://en.wikipedia.org/wiki/Meta-learning_(computer_science)",
        "https://en.wikipedia.org/wiki/Online_machine_learning",
        "https://en.wikipedia.org/wiki/AutoML",
        "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning_control"
    ]

    # Καλούμε τη συνάρτηση
    collect_wikipedia_definitions(
        url_list=wikipedia_pages,
        output_filename="wiki_definitions.json"
    )
