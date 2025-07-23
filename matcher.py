import os
import pickle
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# === Load Assets ===
def load_pickle(file_path: str):
    """Safely load a pickle file from the given path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


try:
    vectorizer: TfidfVectorizer = load_pickle("vectorizer.pkl")
    resume_vectors = load_pickle("resume_vectors.pkl")
    resume_names: List[str] = load_pickle("resume_names.pkl")
except Exception as e:
    print(f"Error during loading pickles: {e}")
    raise


# === Ranking Logic ===
def rank_resumes(job_description: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Rank resumes by cosine similarity to the job description.

    Args:
        job_description (str): The job description text.
        top_k (int): Number of top matches to return.

    Returns:
        List of tuples: (resume_name, similarity_score)
    """
    if not job_description:
        raise ValueError("Job description must not be empty.")

    job_vector = vectorizer.transform([job_description])
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

    # Sort indices by highest score first
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    # Return resume names and their similarity scores
    return [(resume_names[i], float(similarity_scores[i])) for i in top_indices]
