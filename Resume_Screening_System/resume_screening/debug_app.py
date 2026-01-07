import pickle
import pandas as pd
import plotly.graph_objects as go
from utils.text_cleaner import clean_text
from utils.skill_extractor import extract_skills, SKILL_DB

def create_probability_chart(probs, classes):
    """Creates a horizontal bar chart for all class probabilities"""
    df = pd.DataFrame({'Role': classes, 'Probability': probs * 100})
    df = df.nlargest(10, 'Probability')
    df = df.sort_values('Probability', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Role'],
        orientation='h',
        marker=dict(color=df['Probability'], colorscale='Plasma', showscale=False)
    ))
    return fig

def debug_run():
    print("Loading models...")
    try:
        model = pickle.load(open('model/resume_model.pkl', 'rb'))
        tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
    except Exception as e:
        print(f"CRITICAL: Model load failed: {e}")
        return

    sample_text = "Java developer with Spring Boot and SQL experience."
    print(f"Processing text: {sample_text}")

    # 1. Clean
    clean_resume = clean_text(sample_text)
    
    # 2. Skills
    resume_skills = extract_skills(clean_resume)
    print(f"Skills found: {resume_skills}")

    # 3. Predict
    try:
        features = tfidf.transform([clean_resume])
        probs = model.predict_proba(features)[0]
        top_idx = probs.argmax()
        predicted_role = model.classes_[top_idx]
        print(f"Predicted: {predicted_role}")
    except Exception as e:
        print(f"CRITICAL: Prediction failed: {e}")
        return

    # 4. Charts (Simulate)
    try:
        print("Generating charts...")
        create_probability_chart(probs, model.classes_)
        print("Charts generated successfully.")
    except Exception as e:
        print(f"CRITICAL: Chart generation failed: {e}")
        return

    # 5. Skill DB Lookup (Simulate app logic)
    print("Checking Skill DB...")
    all_cat_skills = SKILL_DB.get(predicted_role, [])
    print(f"Skills for '{predicted_role}': {len(all_cat_skills)} found")
    if not all_cat_skills:
        print(f"WARNING: No skills found for category '{predicted_role}'. Check SKILL_DB keys.")

if __name__ == "__main__":
    debug_run()
