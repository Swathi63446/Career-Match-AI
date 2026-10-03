import os
import shutil
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

PERSIST_DIRECTORY = "./vectorstore"

# Comprehensive Skill & Framework Knowledge Base
SKILL_DOCUMENTS = [
    {
        "skill": "Python",
        "category": "Programming Languages",
        "description": "A high-level, general-purpose programming language widely used for web development backends, artificial intelligence, machine learning, data engineering, and automation scripting.",
        "related": "Flask, FastAPI, Django, LangChain, PyTorch, Scikit-Learn, SQL, Pandas",
        "use_cases": "Building REST APIs, developing AI/ML pipelines, writing backend business logic, writing automated testing scripts.",
        "job_requirements": "Requires strong knowledge of Object-Oriented Programming (OOP), async IO, package management (pip/uv), and clean code standards."
    },
    {
        "skill": "Flask",
        "category": "Backend Development",
        "description": "A lightweight WSGI web framework in Python. Used for developing modular backend web applications, microservices, and server-rendered dashboards.",
        "related": "Python, REST APIs, Jinja2, Web Development, FastAPI, Werkzeug, SQLAlchemy",
        "use_cases": "Building lightweight microservices, designing RESTful web APIs, rapid prototyping of backend applications.",
        "job_requirements": "Requires understanding of HTTP request-response cycles, session handling, database ORMs, route authorization, and middleware."
    },
    {
        "skill": "FastAPI",
        "category": "Backend Development",
        "description": "A modern, high-performance Python web framework for building APIs with automatic OpenAPI/Swagger documentation and fast asynchronous request handling.",
        "related": "Python, REST APIs, Pydantic, Asyncio, OpenAPI, Flask, Uvicorn",
        "use_cases": "Building high-throughput microservices, real-time asynchronous API pipelines, schema-validated web endpoints.",
        "job_requirements": "Requires experience with Python type hinting, Pydantic schema design, asynchronous syntax (async/await), and JWT authentication."
    },
    {
        "skill": "Django",
        "category": "Backend Development",
        "description": "A batteries-included, high-level Python web framework that includes built-in admin dashboards, ORMs, authentication systems, and database migration utilities.",
        "related": "Python, SQL, PostgreSQL, REST APIs, Django REST Framework, Web Development",
        "use_cases": "Building full-stack enterprise web portals, secure content management systems, complex web platforms.",
        "job_requirements": "Requires knowledge of Django ORM, migration workflows, security protection (CSRF/XSS), and Django REST Framework (DRF)."
    },
    {
        "skill": "SQL",
        "category": "Databases",
        "description": "Structured Query Language used for defining, querying, joining, and manipulating relational database management systems.",
        "related": "PostgreSQL, MySQL, SQLite, Relational Databases, Database Design, ORM",
        "use_cases": "Data modeling, writing multi-table JOIN queries, indexing data structures, managing transaction ACID compliance.",
        "job_requirements": "Requires ability to write optimized aggregation queries, build relational schemas, design foreign keys, and perform query profiling."
    },
    {
        "skill": "PostgreSQL",
        "category": "Databases",
        "description": "An enterprise-grade, open-source relational database supporting ACID transactions, complex indexing, SQL queries, and native JSON parsing.",
        "related": "SQL, Databases, ORM, Django, Flask, Relational Databases, PGVector",
        "use_cases": "Storing primary application state, executing complex JOIN transactions, storing structured JSON payloads alongside relational data.",
        "job_requirements": "Requires knowledge of index tuning, connection pooling, database migrations, connection security, and query optimization."
    },
    {
        "skill": "MongoDB",
        "category": "Databases",
        "description": "A popular NoSQL document database designed for high availability, horizontal scaling, and flexibility with JSON-like BSON documents.",
        "related": "NoSQL, Databases, JSON, Unstructured Data, Mongoose, Express",
        "use_cases": "Storing unstructured or rapidly evolving schemas, real-time analytics aggregation, distributed document storage.",
        "job_requirements": "Requires experience with document schema design, aggregation pipelines, indexing fields, and replica set cluster management."
    },
    {
        "skill": "Docker",
        "category": "DevOps & Cloud",
        "description": "An open-source containerization technology that packages application source code, runtimes, system tools, and libraries into portable containers.",
        "related": "Containers, Kubernetes, AWS, GCP, Azure, CI/CD, DevOps",
        "use_cases": "Standardizing local development environments, creating multi-container orchestration with Docker Compose, deploying cloud containers.",
        "job_requirements": "Requires writing optimized multi-stage Dockerfiles, managing container environment variables, handling networking ports, and volume mounting."
    },
    {
        "skill": "AWS",
        "category": "Cloud Computing",
        "description": "Amazon Web Services cloud computing platform providing scalable cloud services including virtual servers (EC2), object storage (S3), container runtime (ECS), and serverless execution (Lambda).",
        "related": "Cloud Services, Docker, EC2, S3, Azure, GCP, DevOps, IAM",
        "use_cases": "Deploying web applications to cloud servers, storing media/document files in secure cloud buckets, hosting cloud serverless APIs.",
        "job_requirements": "Requires understanding of IAM access policies, VPC cloud networking, server provisioning, security groups, and cloud storage management."
    },
    {
        "skill": "Azure",
        "category": "Cloud Computing",
        "description": "Microsoft's cloud platform offering cloud virtual machines, managed App Services, enterprise identity management (Entra ID), and cloud database hosting.",
        "related": "Cloud Services, AWS, GCP, DevOps, Enterprise Infrastructure, App Service",
        "use_cases": "Hosting enterprise application platforms, configuring active directory permissions, running hybrid cloud deployments.",
        "job_requirements": "Requires knowledge of Azure Resource Manager templates, Active Directory permissions, App Service deployment, and cloud networking."
    },
    {
        "skill": "GCP",
        "category": "Cloud Computing",
        "description": "Google Cloud Platform delivering infrastructure, container orchestration (GKE), BigQuery data warehouses, and Vertex AI model hosting.",
        "related": "Cloud Services, AWS, Azure, BigQuery, Vertex AI, Kubernetes, DevOps",
        "use_cases": "Hosting large data analytics pipelines, running containerized Kubernetes apps, training and deploying AI/ML models.",
        "job_requirements": "Requires knowledge of Google Cloud IAM roles, Cloud Run serverless containers, cloud storage buckets, and BigQuery SQL querying."
    },
    {
        "skill": "LangChain",
        "category": "AI Frameworks",
        "description": "An open-source software framework engineered to simplify the creation of applications using large language models (LLMs), agents, prompt chains, and vector retrievers.",
        "related": "Gemini API, RAG, ChromaDB, Python, Multi-Agent Systems, Prompt Engineering",
        "use_cases": "Building multi-agent decision workflows, orchestrating Retrieval-Augmented Generation systems, parsing structured LLM responses.",
        "job_requirements": "Requires deep understanding of chain abstractions, output parsing schemas, vector store retrievers, agent memory, and prompt templates."
    },
    {
        "skill": "RAG",
        "category": "AI Systems",
        "description": "Retrieval-Augmented Generation architectural pattern that retrieves relevant information from a vector database to ground LLM context and prevent model hallucinations.",
        "related": "ChromaDB, Vector Databases, LangChain, Gemini API, Semantic Search, Embeddings",
        "use_cases": "Building document question-answering applications, contextual skill matching, search engines grounded in custom knowledge bases.",
        "job_requirements": "Requires experience with text chunking strategies, vector embedding generation, top-K retrieval filtering, and context window assembly."
    },
    {
        "skill": "Machine Learning",
        "category": "Artificial Intelligence",
        "description": "A domain of artificial intelligence focused on building mathematical algorithms that discover patterns from data to perform classification, regression, or clustering.",
        "related": "Python, PyTorch, Scikit-Learn, XGBoost, Deep Learning, Data Science",
        "use_cases": "Predictive modeling, candidate scoring, anomaly detection, automated document categorization, statistical pattern identification.",
        "job_requirements": "Requires understanding of feature engineering, model training workflows, cross-validation, hyperparameter tuning, and performance evaluation metrics."
    },
    {
        "skill": "Power BI",
        "category": "Data Analytics",
        "description": "An enterprise business intelligence dashboarding tool for modeling raw datasets, creating dynamic visual analytics, and calculating metrics using DAX formulas.",
        "related": "SQL, Data Analytics, Dashboards, Excel, Data Modeling, DAX",
        "use_cases": "Designing executive business metrics dashboards, building relational data models, writing DAX measures, visualizing company key indicators.",
        "job_requirements": "Requires experience designing star schema data models, writing DAX expressions, configuring automated data refreshes, and building drill-down charts."
    }
]


def run_ingestion():
    # 1. Clean old vector database to avoid dimension mismatch
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"[Ingest] Removing old vectorstore at '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    # 2. Initialize FastEmbed ONNX embedding model (~120MB RAM, PyTorch-free)
    print("[Ingest] Initializing FastEmbed model 'sentence-transformers/all-MiniLM-L6-v2'...")
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = []

    # 3. Format each record into a LangChain Document with metadata
    for item in SKILL_DOCUMENTS:
        formatted_content = (
            f"Technology Skill: {item['skill']}\n"
            f"Category: {item['category']}\n"
            f"Description: {item['description']}\n"
            f"Related Skills & Ecosystem: {item['related']}\n"
            f"Common Use Cases: {item['use_cases']}\n"
            f"Typical Requirements: {item['job_requirements']}"
        )

        doc = Document(
            page_content=formatted_content,
            metadata={
                "skill_name": item["skill"],
                "category": item["category"],
                "related_skills": item["related"]
            }
        )
        documents.append(doc)

    print(f"[Ingest] Creating persistent ChromaDB database at '{PERSIST_DIRECTORY}'...")

    # 4. Ingest documents into ChromaDB
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"[Ingest] Complete! Successfully embedded {len(documents)} skill documents using FastEmbed.")


if __name__ == "__main__":
    run_ingestion()