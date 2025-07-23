# generate_data.py

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_sample_resumes():
    """
    Loads sample resumes. You can later replace this with a function
    that reads real resumes from files or a database.
    """
    return [
        ("resume_software_engineer.pdf", "Experienced software engineer skilled in Python, JavaScript, and backend systems."),
        ("resume_frontend_developer.pdf", "Frontend developer with expertise in React, CSS, and responsive design."),
        ("resume_data_scientist.pdf", "Data scientist with a background in statistical modeling and machine learning.")
    ]

def save_pickle_file(data, filename):
    """Utility to save data to a .pkl file"""
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved {filename}")

def main():
    # Load resumes
    resumes = load_sample_resumes()

    # Split into names and text content
    resume_names = [name for name, _ in resumes]
    resume_texts = [text for _, text in resumes]

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    resume_vectors = vectorizer.fit_transform(resume_texts)

    # Save artifacts
    save_pickle_file(vectorizer, "vectorizer.pkl")
    save_pickle_file(resume_vectors, "resume_vectors.pkl")
    save_pickle_file(resume_names, "resume_names.pkl")

    print("🎉 All pickle files generated successfully!")

if __name__ == "__main__":
    main()

