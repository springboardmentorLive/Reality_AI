import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import clean_text

# Sample dataset for training
# In a real-world scenario, you would have a larger dataset
sample_data = {
    'resume_text': [
        # Data Scientist resumes
        "Experienced data scientist with expertise in machine learning, deep learning, Python, R, TensorFlow, PyTorch, statistical analysis, data visualization, pandas, numpy, scikit-learn, neural networks, NLP, computer vision",
        "Data scientist skilled in predictive modeling, statistical analysis, Python programming, SQL, data mining, machine learning algorithms, random forest, gradient boosting, feature engineering, A/B testing",
        "PhD in Statistics with 5 years experience in data science, machine learning, Python, R, Spark, Hadoop, big data analytics, data visualization, Tableau, Power BI, regression analysis, classification",
        "Machine learning engineer specializing in deep learning, neural networks, TensorFlow, Keras, Python, computer vision, NLP, data preprocessing, model deployment, MLOps, Docker, Kubernetes",
        "Data scientist with strong background in mathematics, statistics, Python, R, SQL, data analysis, predictive modeling, time series forecasting, clustering, dimensionality reduction, PCA",
        "Senior data scientist experienced in building ML pipelines, feature engineering, model optimization, Python, scikit-learn, XGBoost, LightGBM, data visualization, Jupyter notebooks, Git",
        "Data science professional with expertise in statistical modeling, hypothesis testing, Python, pandas, numpy, matplotlib, seaborn, machine learning, supervised learning, unsupervised learning",
        "AI/ML specialist with experience in deep learning, reinforcement learning, Python, TensorFlow, PyTorch, neural networks, CNN, RNN, LSTM, transformer models, BERT, GPT",
        
        # Java Developer resumes
        "Java developer with 5 years experience in Spring Boot, Spring MVC, Hibernate, REST API, microservices, Maven, Gradle, JUnit, MySQL, PostgreSQL, Git, Agile development",
        "Senior Java engineer skilled in J2EE, Spring Framework, Spring Cloud, Docker, Kubernetes, Jenkins, CI/CD, RESTful web services, SOAP, XML, JSON, multithreading",
        "Full stack Java developer experienced in Spring Boot, Angular, React, JavaScript, TypeScript, HTML, CSS, REST API, MongoDB, MySQL, Git, Agile, Scrum",
        "Java backend developer with expertise in microservices architecture, Spring Boot, Spring Security, OAuth2, JWT, Kafka, RabbitMQ, Redis, Docker, AWS, cloud computing",
        "Java software engineer with strong knowledge of OOP, design patterns, Spring Framework, Hibernate, JPA, SQL, NoSQL, unit testing, integration testing, TDD",
        "Enterprise Java developer experienced in building scalable applications using Spring Boot, Spring Data, Spring Cloud, Netflix OSS, Eureka, Ribbon, Hystrix, Zuul",
        "Java developer proficient in Core Java, J2EE, Servlets, JSP, JDBC, Spring MVC, Hibernate, Maven, Tomcat, WebLogic, database design, SQL optimization",
        "Backend Java engineer with experience in microservices, REST API, GraphQL, Spring Boot, JPA, PostgreSQL, MongoDB, Redis, Docker, Kubernetes, AWS Lambda",
        
        # Web Developer resumes
        "Front-end web developer skilled in HTML5, CSS3, JavaScript, React, Redux, TypeScript, responsive design, Bootstrap, Material-UI, Webpack, npm, Git, Agile",
        "Full stack web developer experienced in React, Node.js, Express, MongoDB, REST API, GraphQL, JavaScript, TypeScript, HTML, CSS, SASS, Git, Docker",
        "Web developer with expertise in modern JavaScript frameworks, React, Vue.js, Angular, HTML5, CSS3, responsive web design, cross-browser compatibility, performance optimization",
        "Senior web developer proficient in React, Next.js, Node.js, Express, PostgreSQL, MongoDB, REST API, authentication, authorization, JWT, OAuth, cloud deployment",
        "UI/UX web developer with strong skills in HTML, CSS, JavaScript, React, Vue.js, Figma, Adobe XD, responsive design, accessibility, SEO, web performance",
        "Full stack JavaScript developer experienced in MERN stack (MongoDB, Express, React, Node.js), REST API, GraphQL, Redux, Webpack, Babel, Jest, testing",
        "Web developer specializing in front-end technologies, React, TypeScript, HTML5, CSS3, SASS, Tailwind CSS, responsive design, progressive web apps, PWA",
        "Modern web developer with expertise in React, Next.js, TypeScript, Node.js, Express, PostgreSQL, Prisma, tRPC, authentication, deployment, Vercel, AWS",
    ],
    'job_role': [
        'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist',
        'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist',
        'Java Developer', 'Java Developer', 'Java Developer', 'Java Developer',
        'Java Developer', 'Java Developer', 'Java Developer', 'Java Developer',
        'Web Developer', 'Web Developer', 'Web Developer', 'Web Developer',
        'Web Developer', 'Web Developer', 'Web Developer', 'Web Developer',
    ]
}

def train_and_save_model():
    """
    Train the resume classification model and save it along with the vectorizer.
    """
    print("Creating dataset...")
    df = pd.DataFrame(sample_data)
    
    print("Cleaning text data...")
    df['cleaned_text'] = df['resume_text'].apply(clean_text)
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['job_role'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['job_role']
    )
    
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_tfidf, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nSaving model and vectorizer...")
    with open('resume_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model and vectorizer saved successfully!")
    print("Files created: resume_classifier.pkl, tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_and_save_model()
