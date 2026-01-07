import pandas as pd
import numpy as np
import pickle
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from utils.text_cleaner import clean_text

# Define categories
CATEGORIES = ['Data Scientist', 'Web Developer', 'Java Developer', 'Python Developer', 'HR', 'DevOps', 'Other']

def load_real_data(filepath='ml_resume_dataset_4500.csv'):
    """
    Loads real resume data and maps it to simplified categories.
    Mapping Strategy:
    - Titles containing 'Data', 'Machine Learning', 'AI', 'NLP' -> 'Data Scientist'
    - All other titles -> 'Other' (Non-Technical/General)
    """
    try:
        df = pd.read_csv(filepath)
        
        # We only care about the text and the label/category
        data = []
        labels = []
        
        for _, row in df.iterrows():
            title = str(row['current_title']).lower()
            text = str(row['raw_text']) # assuming 'raw_text' column exists based on file inspection
            
            # Simple keyword-based mapping for the real dataset
            if any(x in title for x in ['data', 'machine', 'learning', 'ai', 'nlp', 'vision', 'statistic', 'research']):
                category = 'Data Scientist'
            else:
                category = 'Other'
            
            data.append(clean_text(text))
            labels.append(category)
            
        print(f"Loaded {len(data)} real resumes.")
        return data, labels
    except Exception as e:
        print(f"Error loading real data: {e}")
        return [], []

def generate_synthetic_data(num_samples_per_category=200):
    """
    Generates a rich synthetic dataset for training.
    """
    print(f"Generating {num_samples_per_category} synthetic resumes per category...")
    data = []
    
    # Richer keyword banks with phrases - EXPERIMENTAL FIX FOR BIAS
    # Made keywords more distinct to prevent Data Scientist overlap
    keyword_bank = {
        'Data Scientist': [
            'machine learning', 'deep learning', 'scikit-learn', 'statistics', 'random forest', 'xgboost',
            'data analysis', 'visualization', 'matplotlib', 'seaborn', 'tensorflow', 'keras', 'pytorch', 'nlp', 
            'natural language processing', 'computer vision', 'predictive modeling', 'a/b testing', 'big data', 'spark', 
            'neural networks', 'linear regression', 'logistic regression', 'k-means', 'clustering'
        ],
        'Web Developer': [
            'html', 'html5', 'css', 'css3', 'javascript', 'react', 'react.js', 'angular', 'vue.js', 'node.js', 
            'frontend', 'backend', 'full stack', 'api', 'rest api', 'responsive design', 'bootstrap', 'tailwind',
            'jquery', 'ajax', 'json', 'xml', 'ui/ux', 'web design', 'figma', 'webpack', 'npm', 'dom manipulation'
        ],
        'Java Developer': [
            'java', 'core java', 'j2ee', 'spring', 'spring boot', 'hibernate', 'jpa', 'maven', 'gradle', 
            'junit', 'mockito', 'oop', 'object oriented programming', 'mvc', 'microservices', 'struts', 'jsp', 'servlet',
            'jdbc', 'multithreading', 'concurrency', 'design patterns', 'eclipse', 'intellij', 'jvm'
        ],
        'Python Developer': [
            'python', 'django', 'flask', 'fastapi', 'scripting', 'automation', 'selenium', 'beautifulsoup', 
            'requests', 'sqlalchemy', 'pep8', 'unit testing', 'pytest', 'celery', 'redis', 'gunicorn',
            'backend', 'api development', 'rest', 'endpoints', 'server-side'
        ],
        'HR': [
            'human resources', 'recruitment', 'talent acquisition', 'sourcing', 'screening', 'interviewing', 
            'hiring', 'onboarding', 'employee relations', 'performance management', 'payroll', 'compensation', 
            'benefits', 'labor laws', 'compliance', 'hris', 'workday', 'bamboo hr', 'culture', 'training', 'development'
        ],
        'DevOps': [
            'devops', 'ci/cd', 'continuous integration', 'continuous deployment', 'jenkins', 'gitlab ci', 'docker', 
            'kubernetes', 'k8s', 'aws', 'amazon web services', 'azure', 'gcp', 'cloud', 'infrastructure as code', 
            'terraform', 'ansible', 'chef', 'puppet', 'linux', 'bash', 'shell scripting', 'monitoring', 'nagios'
        ],
        'Other': [
            'medical', 'doctor', 'nurse', 'patient', 'healthcare', 'clinic', 'pharmacy', 'medicine', 'hospital',
            'teacher', 'education', 'classroom', 'student', 'lesson', 'curriculum', 'teaching', 'school',
            'sales', 'customer service', 'retail', 'manager', 'store', 'cashier', 'inventory', 'client',
            'driver', 'truck', 'delivery', 'logistics', 'warehouse', 'transport',
            'chef', 'cook', 'kitchen', 'restaurant', 'food', 'culinary', 'menu',
            'receptionist', 'clerk', 'office', 'assistant', 'telephone', 'filing', 'admin'
        ]
    }
    
    # Templates to make text look more like sentences
    templates = [
        "Experienced in {}.", "Proficient with {}.", "Skilled in usage of {}.", 
        "Completed projects involves {}.", "Expertise includes {}.", "Hands-on experience with {}.",
        "Knowledge of {}.", "Implemented various solutions using {}.", "Worked extensively on {}."
    ]

    for category in CATEGORIES:
        base_keywords = keyword_bank[category]
        
        for i in range(num_samples_per_category):
            # Pick a random subset of keywords
            num_keywords = random.randint(10, 20)
            selected_keywords = random.choices(base_keywords, k=num_keywords)
            
            # Construct a fake resume text
            resume_text = f"Resume for {category} role. "
            
            # Add some sentence-like structures
            for _ in range(3):
                phrase = random.choice(templates).format(", ".join(random.sample(selected_keywords, 3)))
                resume_text += phrase + " "
            
            # Add a raw dump of keywords for density
            resume_text += " Skills: " + " ".join(selected_keywords)
            
            data.append({'Resume': resume_text, 'Category': category})
            
    return pd.DataFrame(data)

def train_model():
    print("Loading Real Data...")
    real_data, real_labels = load_real_data()

    print("Generating Synthetic Data (to supplement missing roles)...")
    # Generate synthetic data for other roles to ensure balance
    df_syn = generate_synthetic_data(num_samples_per_category=200)
    syn_data = df_syn['Cleaned_Resume'].tolist() if 'Cleaned_Resume' in df_syn.columns else [clean_text(x) for x in df_syn['Resume']]
    syn_labels = df_syn['Category'].tolist()

    # Combine datasets
    X_train_text = real_data + syn_data
    y_train_labels = real_labels + syn_labels
    
    print(f"Total training samples: {len(X_train_text)}")
    
    from collections import Counter
    print("Training distribution:")
    print(Counter(y_train_labels))

    # X_train_clean is already clean from previous steps
    X_train_clean = X_train_text 

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_clean, y_train_labels, test_size=0.2, random_state=42, stratify=y_train_labels
    )
    
    # 4. Feature Extraction (TF-IDF)
    print("Vectorizing...")
    # max_df=0.85: Ignore terms that appear in >85% of documents (too common)
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), max_df=0.85)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 5. Model Training
    print("Training K-Nearest Neighbors...")
    # KNN is good for finding the "closest match" in the synthetic examples
    # metric='cosine' is standard for text similarity
    clf = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    clf.fit(X_train_tfidf, y_train)
    
    # 6. Evaluation
    preds = clf.predict(X_test_tfidf)
    print(" Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    # 7. Save Artifacts
    if not os.path.exists('model'):
        os.makedirs('model')
        
    print("Saving model and vectorizer to model/ folder...")
    pickle.dump(clf, open('model/resume_model.pkl', 'wb'))
    pickle.dump(tfidf, open('model/tfidf.pkl', 'wb'))
    
    print("Done! Model retrained on high-quality synthetic data.")

if __name__ == "__main__":
    train_model()
