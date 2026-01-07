import re

# Defined skill set for various domains
SKILL_DB = {
    'Data Science': [
        'python', 'r', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'matplotlib', 'seaborn',
        'tensorflow', 'keras', 'pytorch', 'sql', 'tableau', 'power bi', 'statistics',
        'machine learning', 'deep learning', 'nlp', 'data analysis', 'big data', 'hadoop', 'spark'
    ],
    'Data Scientist': [
        'python', 'r', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'matplotlib', 'seaborn',
        'tensorflow', 'keras', 'pytorch', 'sql', 'tableau', 'power bi', 'statistics',
        'machine learning', 'deep learning', 'nlp', 'data analysis', 'big data', 'hadoop', 'spark'
    ],
    'Senior Data Scientist': [
        'python', 'r', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'matplotlib', 'seaborn',
        'tensorflow', 'keras', 'pytorch', 'sql', 'tableau', 'power bi', 'statistics',
        'machine learning', 'deep learning', 'nlp', 'data analysis', 'big data', 'hadoop', 'spark', 'leadership', 'communication'
    ],
    'Machine Learning Engineer': [
        'python', 'c++', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'sql', 
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'mlops', 'flask', 'fastapi', 'git'
    ],
    'Applied ML Engineer': [
        'python', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'sql', 
        'docker', 'kubernetes', 'mlops', 'deployment'
    ],
    'Computer Vision Engineer': [
        'python', 'opencv', 'pytorch', 'tensorflow', 'keras', 'image processing', 'deep learning',
        'cnn', 'yolo', 'object detection', 'segmentation'
    ],
    'NLP Engineer': [
        'python', 'nltk', 'spacy', 'transformers', 'huggingface', 'bert', 'gpt', 'lstm', 'rnn',
        'text processing', 'pytorch', 'tensorflow'
    ],
    'AI Researcher': [
        'python', 'pytorch', 'tensorflow', 'deep learning', 'research', 'statistics', 'mathematics',
        'latex', 'publication'
    ],
    'Data Analyst': [
        'excel', 'sql', 'python', 'r', 'tableau', 'power bi', 'statistics', 'data visualization',
        'analysis', 'cleaning'
    ],
    'Web Developer': [
        'html', 'css', 'javascript', 'js', 'react', 'angular', 'vue', 'node.js', 'node',
        'express', 'django', 'flask', 'php', 'laravel', 'sql', 'mysql', 'mongodb', 
        'postgresql', 'api', 'rest', 'graphql', 'bootstrap', 'tailwind', 'jquery'
    ],
    'Java Developer': [
        'java', 'spring', 'spring boot', 'hibernate', 'j2ee', 'jsp', 'servlets', 'jdbc',
        'maven', 'gradle', 'junit', 'mockito', 'microservices', 'soap', 'rest', 'sql'
    ],
    'Python Developer': [
        'python', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'scipy', 'pytest',
        'selenium', 'beautifulsoup', 'scrapy', 'sqlalchemy', 'celery', 'redis'
    ],
    'DevOps': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
        'ansible', 'terraform', 'linux', 'bash', 'shell', 'nginx', 'apache', 'monitoring', 
        'prometheus', 'grafana'
    ],
    'HR': [
        'recruitment', 'talent acquisition', 'sourcing', 'screening', 'interviewing',
        'hiring', 'onboarding', 'employee relations', 'compliance', 'communication',
        'negotiation', 'leadership', 'management'
    ],
    'Customer Support': [
        'communication', 'crm', 'salesforce', 'zendesk', 'empathy', 'problem solving',
        'ticketing', 'phone skills', 'chat support'
    ],
    'Telecaller': [
        'communication', 'cold calling', 'sales', 'persuasion', 'customer service',
        'record keeping', 'telemarketing'
    ],
    'Receptionist': [
        'communication', 'organization', 'scheduling', 'microsoft office', 'excel', 'word',
        'phone etiquette', 'filing', 'front desk'
    ],
    'Sales Executive': [
        'sales', 'marketing', 'negotiation', 'crm', 'lead generation', 'communication',
        'presentation', 'closing'
    ],
    'Data Entry Operator': [
        'typing', 'excel', 'word', 'data entry', 'accuracy', 'attention to detail',
        '10-key', 'spreadsheets'
    ],
    'Administrative Assistant': [
        'scheduling', 'organization', 'microsoft office', 'filing', 'communication',
        'email management'
    ],
    'Office Assistant': [
        'organization', 'filing', 'copying', 'microsoft office', 'communication', 'clerical'
    ],
    'Content Writer': [
        'writing', 'editing', 'seo', 'blogging', 'copywriting', 'content strategy',
        'research', 'proofreading', 'wordpress'
    ],
    'Marketing Intern': [
        'social media', 'content creation', 'marketing', 'seo', 'canva', 'communication',
        'analytics'
    ],
    'Junior Intern': [
        'communication', 'learning', 'teamwork', 'basic computer skills', 'microsoft office'
    ]
}

def extract_skills(text):
    """
    Extracts skills from the provided text based on the SKILL_DB.
    Returns a list of unique found skills.
    """
    found_skills = set()
    # Normalize text
    text = text.lower()
    
    # Iterate through all categories and skills
    for category, skills in SKILL_DB.items():
        for skill in skills:
            # Match whole words only to avoid false positives (e.g. 'c' in 'act')
            # Escape skill for regex special characters (like . in node.js)
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found_skills.add(skill)
                
    return list(found_skills)
