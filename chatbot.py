import os
import json
import re
import nltk
import numpy as np
import faiss
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
import time
# ------------------------
# Setup NLTK
# ------------------------
#nltk.download("stopwords", quiet=True)
#nltk.download("wordnet", quiet=True)
#nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer=PorterStemmer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    return " ".join([stemmer.stem(lemmatizer.lemmatize(w)) for w in words if w not in stop_words])

class Chatbot:
    def __init__(self, filename="queries.json"):
        self.filename = filename
        self.queries = self.load_data()

        # Models
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Cache
        self.tfidf_matrix = None
        self.embeddings = None
        self.faiss_index = None
        self.dim = 384  # embedding size for MiniLM

        self._update_caches()

    # ------------------------
    # Data Handling
    # ------------------------
    def load_data(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_data(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.queries, f, indent=4)

    # ------------------------
    # Cache Updates
    # ------------------------
    def _update_caches(self):
        if not self.queries:
            self.tfidf_matrix, self.embeddings, self.faiss_index = None, None, None
            return

        keys = list(self.queries.keys())

        # TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(keys)

        # Embeddings
        self.embeddings = self.embedding_model.encode(keys, convert_to_numpy=True)

        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.dim)  # inner product (cosine)
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)

    # ------------------------
    # Matching
    # ------------------------
    def match(self, user_query, threshold=0.65):
        if not self.queries:
            return None, None, 0

        norm_query = preprocess(user_query)

        # TF-IDF score
        query_vec = self.vectorizer.transform([norm_query])
        tfidf_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        tfidf_idx = tfidf_sim.argmax()
        tfidf_score = tfidf_sim[tfidf_idx]

        # Embedding score via FAISS
        user_emb = self.embedding_model.encode([norm_query], convert_to_numpy=True)
        faiss.normalize_L2(user_emb)
        distances, indices = self.faiss_index.search(user_emb, k=1)
        embed_idx = indices[0][0]
        embed_score = float(distances[0][0])

        # Hybrid scoring
        final_key = list(self.queries.keys())[embed_idx if embed_score >= tfidf_score else tfidf_idx]
        final_score = 0.7 * embed_score + 0.3 * tfidf_score

        if final_score >= threshold:
            return self.queries[final_key]["responses"], final_key, final_score
        return None, None, final_score

    def suggest(self, user_query, top_k=3):
        if not self.queries:
            return []
        norm_query = preprocess(user_query)
        user_emb = self.embedding_model.encode([norm_query], convert_to_numpy=True)
        faiss.normalize_L2(user_emb)
        distances, indices = self.faiss_index.search(user_emb, k=top_k)
        return [list(self.queries.keys())[i] for i in indices[0]]

    # ------------------------
    # Learning
    # ------------------------
    def learn(self, user_query, response):
        norm_query = preprocess(user_query)
        if norm_query in self.queries:
            self.queries[norm_query]["responses"].append(response)
            self.queries[norm_query]["responses"] = list(set(self.queries[norm_query]["responses"]))
        else:
            self.queries[norm_query] = {"original": user_query, "responses": [response]}
        self.save_data()
        self._update_caches()
