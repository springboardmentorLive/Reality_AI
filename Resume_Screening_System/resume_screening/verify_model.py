import pickle
import sys
from utils.text_cleaner import clean_text

def verify():
    print("Loading model...")
    try:
        model = pickle.load(open('model/resume_model.pkl', 'rb'))
        tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test cases: (Text, Expected Role)
    test_cases = [
        ("Expert in Java, Spring Boot, Hibernate, microservices, and SQL.", "Java Developer"),
        ("Experienced HR manager with skills in recruitment, employee relations, and payroll.", "HR"),
        ("Frontend developer skilled in React, HTML, CSS, JavaScript, and Node.js.", "Web Developer"),
        ("DevOps engineer with experience in Docker, Kubernetes, AWS, and CI/CD pipelines.", "DevOps"),
        ("Data scientist with strong background in machine learning, python, pandas, and statistics.", "Data Scientist"),
        ("Python developer with Django and Flask experience, writing scripts and automation.", "Python Developer"),
        ("Medical receptionist with experience in scheduling, patient greeting, phone management, and filing.", "Other")
    ]

    print("\n--- Model Verification ---")
    correct = 0
    for text, expected in test_cases:
        cleaned = clean_text(text)
        features = tfidf.transform([cleaned])
        probs = model.predict_proba(features)[0]
        
        # Show top 3 for debug
        top_3_idx = probs.argsort()[-3:][::-1]
        print(f"Input: '{text[:50]}...'")
        for idx in top_3_idx:
            role = model.classes_[idx]
            prob = probs[idx]
            print(f"  - {role}: {prob*100:.1f}%")

        top_idx = probs.argmax()
        prediction = model.classes_[top_idx]
        confidence = probs[top_idx] * 100
        
        status = "✅" if prediction == expected else "❌"
        if prediction == expected: correct += 1
        
        print(f"   Result: {status} {prediction}")
        print("-" * 30)

    print(f"\nScore: {correct}/{len(test_cases)}")
    if correct == len(test_cases):
        print("SUCCESS: Model is predicting diverse roles correctly.")
    else:
        print("WARNING: Model still shows signs of bias or inaccuracy.")

if __name__ == "__main__":
    verify()
