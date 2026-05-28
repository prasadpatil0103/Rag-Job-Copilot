import json
import re
import boto3
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

S3_BUCKET = "rag-job-copilot-jds"
TOP_N_JOBS = 5

SKILL_ALIASES = {
    "pyspark": "spark",
    "apache spark": "spark",
    "apache kafka": "kafka",
    "apache airflow": "airflow",
    "sklearn": "scikit-learn",
    "huggingface": "hugging face",
    "transformers": "hugging face",
    "aws cloud": "aws",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "ms azure": "azure",
    "microsoft azure": "azure",
    "postgres": "postgresql",
    "nlp": "natural language processing",
    "powerbi": "power bi",
    "power-bi": "power bi",
    "structured query language": "sql",
    "k8s": "kubernetes",
    "cicd": "ci/cd",
    "github actions": "ci/cd",
    "tensorflow 2": "tensorflow",
    "pytorch lightning": "pytorch",
}

SKILLS = [
    # Languages
    "python", "r", "sql", "scala", "java", "bash",
    # ML / AI Libraries
    "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost",
    "lightgbm", "hugging face", "spacy", "nltk", "opencv",
    # Data Libraries
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    # ML Concepts
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "regression", "classification",
    "clustering", "time series", "forecasting", "a/b testing", "statistics",
    "statistical modeling", "feature engineering", "model deployment", "mlops", "mlflow",
    # Data Engineering
    "spark", "kafka", "airflow", "dbt", "etl", "elt", "data pipeline",
    "data lake", "data warehouse", "data modeling", "flink", "hadoop", "hive", "streaming",
    # Databases
    "snowflake", "redshift", "bigquery", "postgresql", "mysql",
    "mongodb", "cassandra", "dynamodb", "elasticsearch",
    # Cloud
    "aws", "google cloud", "azure", "s3", "lambda", "ec2", "glue", "sagemaker",
    "emr", "athena", "cloudformation", "terraform", "iam",
    # BI / Visualization
    "tableau", "power bi", "looker", "google analytics", "excel", "quicksight",
    # DevOps / Infra
    "docker", "kubernetes", "git", "ci/cd", "jenkins", "linux", "rest api", "microservices",
]


def detect_role(filename: str) -> str:
    f = filename.lower()
    if f.startswith("ds"):
        return "Data Scientist"
    elif f.startswith("da"):
        return "Data Analyst"
    elif f.startswith("de"):
        return "Data Engineer"
    return "Unknown"


def detect_seniority(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["senior", "sr.", "sr ", "lead", "principal", "staff"]):
        return "Senior"
    elif any(w in t for w in ["junior", "jr.", "entry", "associate", "intern"]):
        return "Entry"
    return "Mid-level"


def normalize_text(text: str) -> str:
    t = text.lower()
    for alias, canonical in SKILL_ALIASES.items():
        t = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, t)
    return t


def extract_skills(text: str) -> list:
    normalized = normalize_text(text)
    return [s for s in SKILLS if re.search(r"\b" + re.escape(s) + r"\b", normalized)]


def parse_metadata(text: str) -> tuple:
    title, company, location = "Unknown", "Unknown", "Unknown"
    for line in text.strip().splitlines()[:5]:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("company:"):
            company = line.split(":", 1)[1].strip()
        elif line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
    return title, company, location


def load_and_parse_jobs(bucket: str) -> list:
    s3 = boto3.client("s3")
    jobs = []
    for obj in s3.list_objects_v2(Bucket=bucket).get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".txt"):
            continue
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        title, company, location = parse_metadata(body)
        jobs.append({
            "filename": key,
            "title": title,
            "company": company,
            "location": location,
            "role": detect_role(key),
            "seniority": detect_seniority(body),
            "skills": extract_skills(body),
            "text": body,
        })
    return jobs


def save_processed_jobs(bucket: str, jobs: list) -> None:
    s3 = boto3.client("s3")
    exportable = [{k: v for k, v in j.items() if k != "text"} for j in jobs]
    s3.put_object(
        Bucket=bucket,
        Key="processed/processed_jobs.json",
        Body=json.dumps(exportable, indent=2),
        ContentType="application/json",
    )


def _response(status: int, data: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": json.dumps(data),
    }


def lambda_handler(event, context):
    # Parse request body
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query", "").strip()
    except Exception:
        return _response(400, {"error": "Invalid request body"})

    if not query:
        return _response(400, {"error": "Query cannot be empty"})

    # Load and parse all job descriptions from S3
    jobs = load_and_parse_jobs(S3_BUCKET)
    if not jobs:
        return _response(500, {"error": "No job descriptions found in S3"})

    # Cache processed jobs back to S3 for the analytics dashboard
    try:
        save_processed_jobs(S3_BUCKET, jobs)
    except Exception as e:
        print(f"Warning: could not save processed jobs: {e}")

    # TF-IDF vectorization + cosine similarity
    texts = [normalize_text(j["text"]) for j in jobs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([normalize_text(query)])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:TOP_N_JOBS]

    # Build response payload
    top_jobs = []
    all_skills = []
    for idx in top_indices:
        j = jobs[idx]
        all_skills.extend(j["skills"])
        top_jobs.append({
            "filename": j["filename"],
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "role": j["role"],
            "seniority": j["seniority"],
            "score": round(float(scores[idx]), 4),
            "skills": j["skills"],
            "snippet": j["text"][:300].replace("\n", " "),
        })

    top_skills = [s for s, _ in Counter(all_skills).most_common(15)]
    return _response(200, {
        "query": query,
        "top_skills": top_skills,
        "top_jobs": top_jobs,
    })
