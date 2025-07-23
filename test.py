from matcher import rank_resumes

def test_match(job_description, top_k=3):
    matches = rank_resumes(job_description, top_k)
    
    if not matches:
        print("⚠️ No matches found.")
        return

    print(f"\n🔍 Job: {job_description}")
    for name, score in matches:
        print(f"   {name}: {score:.4f}")

def main():
    job_descriptions = [
        "Looking for a backend engineer with Python and FastAPI experience.",
        "Frontend role requiring React and UI/UX design.",
        "Need a data scientist with experience in ML and Python."
    ]
    
    for jd in job_descriptions:
        test_match(jd)

if __name__ == "__main__":
    main()
