# data/sample_data.py
# ─────────────────────────────────────────────────────────────
#  10 Candidate Resumes + 3 Job Descriptions
#  Used for RAG Resume Screener 
# ─────────────────────────────────────────────────────────────

RESUMES = [
    {
        "candidate": "Alice Johnson",
        "text": """
        Alice Johnson | alice@email.com | LinkedIn: alice-johnson
        Summary: Experienced NLP Engineer with 4 years in text classification, named entity recognition,
        and transformer-based models. Proficient in Python, HuggingFace Transformers, spaCy, BERT, GPT.
        Skills: Python, NLP, BERT, HuggingFace, spaCy, PyTorch, TensorFlow, scikit-learn, NLTK,
                text classification, NER, sentiment analysis, Docker, REST API, Git
        Experience:
          - NLP Engineer @ TechCorp (2021–2024): Built BERT-based text classifiers, deployed on AWS.
          - Data Scientist @ StartupAI (2020–2021): Developed sentiment analysis pipelines.
        Education: M.Tech in Computer Science, IIT Delhi (2020)
        Projects: Resume Parser, Chatbot with Rasa, Fake News Detector
        """
    },
    {
        "candidate": "Bob Smith",
        "text": """
        Bob Smith | bob@email.com
        Summary: Machine Learning Engineer passionate about deep learning and computer vision.
        Skills: Python, PyTorch, TensorFlow, Keras, CNN, YOLO, OpenCV, scikit-learn,
                pandas, NumPy, Docker, Kubernetes, AWS, MLflow, Git
        Experience:
          - ML Engineer @ VisionTech (2022–2024): Built YOLO-based object detection for retail.
          - Junior Data Scientist @ DataHub (2020–2022): Worked on tabular ML models.
        Education: B.Tech in Electronics, NIT Trichy (2020)
        Projects: Face Detection System, Sales Forecasting, Image Segmentation
        """
    },
    {
        "candidate": "Carol White",
        "text": """
        Carol White | carol@email.com
        Summary: Data Scientist with strong analytics and machine learning background.
        Skills: Python, R, SQL, scikit-learn, XGBoost, LightGBM, pandas, NumPy,
                Tableau, Power BI, statistics, A/B testing, regression, classification
        Experience:
          - Data Scientist @ AnalyticsInc (2021–2024): Led customer churn prediction projects.
          - Business Analyst @ RetailCo (2019–2021): Created dashboards and KPI reports.
        Education: M.Sc Statistics, Delhi University (2019)
        Projects: Customer Churn Predictor, Revenue Forecasting Dashboard
        """
    },
    {
        "candidate": "David Kumar",
        "text": """
        David Kumar | david@email.com
        Summary: Full-stack developer with interest in AI/ML integrations and REST APIs.
        Skills: Python, JavaScript, React, Node.js, FastAPI, Flask, SQL, MongoDB,
                Docker, Git, REST API, HTML, CSS, basic scikit-learn
        Experience:
          - Software Engineer @ WebDev Co (2021–2024): Built full-stack SaaS applications.
          - Intern @ StartupXYZ (2020): Developed REST APIs with Flask.
        Education: B.Tech Computer Science, VIT Vellore (2020)
        Projects: E-commerce Platform, REST API Gateway, Portfolio Website
        """
    },
    {
        "candidate": "Eva Patel",
        "text": """
        Eva Patel | eva@email.com
        Summary: NLP researcher specializing in multilingual models, RAG systems, and LLM fine-tuning.
        Skills: Python, HuggingFace, BERT, GPT, LLaMA, RAG, FAISS, LangChain, spaCy,
                PyTorch, NLP, text generation, embeddings, vector search, fine-tuning, Git
        Experience:
          - NLP Researcher @ AI Lab (2022–2024): Developed RAG pipelines for enterprise Q&A.
          - ML Intern @ DeepMind India (2021): Fine-tuned multilingual BERT.
        Education: M.Tech AI, IIT Bombay (2022)
        Projects: Multilingual Chatbot, RAG Document Search, LLM Fine-Tuning Pipeline
        """
    },
    {
        "candidate": "Frank Lee",
        "text": """
        Frank Lee | frank@email.com
        Summary: Data engineer focused on pipelines, ETL, and big data processing.
        Skills: Python, Spark, Hadoop, Kafka, Airflow, SQL, PostgreSQL, AWS S3, GCP,
                Docker, Kubernetes, dbt, pandas, ETL, data warehousing
        Experience:
          - Data Engineer @ BigDataCo (2020–2024): Built Spark-based ETL pipelines.
          - Backend Engineer @ CloudSoft (2018–2020): Managed PostgreSQL databases.
        Education: B.Tech IT, BITS Pilani (2018)
        Projects: Real-time Data Pipeline, Data Lake Architecture
        """
    },
    {
        "candidate": "Grace Chen",
        "text": """
        Grace Chen | grace@email.com
        Summary: AI researcher with focus on reinforcement learning and robotics.
        Skills: Python, PyTorch, reinforcement learning, OpenAI Gym, ROS, C++,
                deep learning, policy gradient, Q-learning, scikit-learn, Linux
        Experience:
          - RL Researcher @ RoboLab (2022–2024): Developed RL agents for robot navigation.
          - Research Intern @ IISc (2021): Implemented DQN for Atari games.
        Education: M.Tech Robotics, IISc Bangalore (2022)
        Projects: Autonomous Robot Navigation, RL Trading Agent
        """
    },
    {
        "candidate": "Henry Nair",
        "text": """
        Henry Nair | henry@email.com
        Summary: Cloud architect and DevOps engineer with ML deployment experience.
        Skills: AWS, GCP, Azure, Docker, Kubernetes, Terraform, CI/CD, Python,
                MLflow, SageMaker, REST API, Linux, Bash, Git, monitoring
        Experience:
          - Cloud Architect @ CloudPro (2019–2024): Architected ML platforms on AWS.
          - DevOps Engineer @ SoftHouse (2017–2019): Managed CI/CD pipelines.
        Education: B.Tech Computer Science, MNIT Jaipur (2017)
        Projects: ML Platform on AWS, Kubernetes ML Deployment, Cost Optimization Pipeline
        """
    },
    {
        "candidate": "Isla Roy",
        "text": """
        Isla Roy | isla@email.com
        Summary: Junior data scientist, fresh graduate with strong Python and ML fundamentals.
        Skills: Python, scikit-learn, pandas, NumPy, matplotlib, seaborn, SQL,
                regression, classification, clustering, Jupyter, Git
        Experience:
          - Data Science Intern @ AnalyticsCo (2023): Built classification models for HR data.
        Education: B.Tech Computer Science, Jadavpur University (2023)
        Projects: Iris Classification, House Price Prediction, Student Performance Analysis
        """
    },
    {
        "candidate": "Jake Morris",
        "text": """
        Jake Morris | jake@email.com
        Summary: Cybersecurity specialist with some ML experience in anomaly detection.
        Skills: Python, network security, intrusion detection, anomaly detection,
                scikit-learn, Wireshark, Kali Linux, SIEM, penetration testing, Git
        Experience:
          - Security Analyst @ SecureFirm (2020–2024): Built ML-based anomaly detection systems.
          - IT Support @ TechBase (2018–2020): Managed network infrastructure.
        Education: B.Tech IT, Amity University (2018)
        Projects: Network Anomaly Detector, Phishing URL Classifier
        """
    },
]


JOB_DESCRIPTIONS = {
    "NLP Engineer": """
        We are looking for an experienced NLP Engineer to join our AI team.
        Requirements:
        - Strong Python programming skills
        - Experience with HuggingFace Transformers, BERT, GPT models
        - Proficiency in spaCy, NLTK for text preprocessing
        - Knowledge of NLP tasks: text classification, NER, sentiment analysis
        - Experience with PyTorch or TensorFlow
        - Familiarity with REST API and Docker for deployment
        - Good understanding of embeddings and vector search (FAISS, ChromaDB)
        - Bonus: experience with RAG pipelines and LangChain
    """,
    "Data Scientist": """
        Hiring a Data Scientist to drive business insights and predictive modeling.
        Requirements:
        - Proficient in Python and R for data analysis
        - Strong knowledge of ML algorithms: regression, classification, clustering
        - Experience with scikit-learn, XGBoost, LightGBM
        - SQL proficiency and data wrangling with pandas
        - Data visualization using Tableau, Power BI, matplotlib
        - Understanding of statistics, A/B testing, hypothesis testing
        - Bonus: experience with cloud platforms (AWS, GCP)
    """,
    "Machine Learning Engineer": """
        Seeking a Machine Learning Engineer to build and deploy ML models at scale.
        Requirements:
        - Expert Python skills with PyTorch or TensorFlow
        - Experience deploying models using Docker, Kubernetes
        - Familiarity with MLflow or similar experiment tracking tools
        - Knowledge of deep learning architectures (CNN, RNN, Transformers)
        - Cloud experience: AWS SageMaker or GCP Vertex AI
        - REST API development with FastAPI or Flask
        - Strong Git and CI/CD practices
        - Bonus: experience with model optimization and quantization
    """
}
