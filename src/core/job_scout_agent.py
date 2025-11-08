import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# --- 1. CONFIGURATION ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
_model_cache = None # Cache variable to hold the model after first load

# Base list of companies with detailed descriptions (simulating web data aggregation)
CONCEPTUAL_COMPANIES = {
    "Lahore": [
        {"name": "Techlogix", "focus": "Enterprise Software, Python APIs, Data Science Consulting.", "status": "Hiring (SE-Python)", "link": "https://techlogix.com"},
        {"name": "Systems Limited", "focus": "Digital Transformation, AI/ML Services, Cloud, Fintech.", "status": "Hiring (Senior AI)", "link": "https://syslmt.com"},
        {"name": "Arbisoft", "focus": "Custom Software Development, MLOps, CV/ML solutions.", "status": "No Open Roles", "link": "https://arbisoft.com"},
        {"name": "NetSol Technologies", "focus": "Leasing and Finance Software, Data Analytics.", "status": "Hiring (Junior SE)", "link": "https://netsoltech.com"},
        {"name": "Confiz", "focus": "E-commerce and Retail Solutions, Data Engineering.", "status": "Hiring (DevOps)", "link": "https://confiz.com"},
    ],
    "Islamabad": [
        {"name": "Afiniti", "focus": "Behavioral Science AI, Large-scale ML Algorithms, CloudOps.", "status": "Hiring (ML Engineer)", "link": "https://afiniti.com"},
        {"name": "VentureDive", "focus": "Digital Products, Data Engineering, Mobile App Development.", "status": "Hiring (Senior SE)", "link": "https://venturedive.com"},
        {"name": "Contour Software", "focus": "Global Software R&D, Financial Tech, Data Services.", "status": "No Open Roles", "link": "https://contour.com.pk"},
        {"name": "Teradata PK", "focus": "Data Warehousing, Data Analytics, BI Solutions.", "status": "Hiring (Data Analyst)", "link": "https://teradata.com"},
    ],
    "Mianwali": [
        {"name": "NiaziTech Hub", "focus": "Local AI Service Development, Community Tech Training.", "status": "Hiring (SE - Intern)", "link": "https://niazitech.pk"},
        {"name": "Rural Tech Solutions", "focus": "AgriTech and Local Data Solutions using Python.", "status": "Hiring (Junior Data)", "link": "https://ruraltech.org"},
    ]
}

# --- 2. LAZY MODEL LOADING FUNCTION ---

def _get_model():
    """Loads the S-Transformer model only once, using caching."""
    global _model_cache
    if _model_cache is None:
        print(f"--- Initializing Sentence Transformer Model ({EMBEDDING_MODEL}) ---")
        try:
            # Model initialization happens here (slowest part, forces download if missing)
            _model_cache = SentenceTransformer(EMBEDDING_MODEL)
            print("--- Model Loaded Successfully ---")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load SentenceTransformer. Check internet/dependencies. Error: {e}")
            raise e
    return _model_cache

# --- 3. SEMANTIC RANKING LOGIC ---

def get_tech_industry_data(city: str, profile_text: str) -> pd.DataFrame:
    """
    Ranks local companies based on semantic match with the candidate's profile.
    """
    try:
        # Load model only when the function is called
        model = _get_model()
    except Exception:
        # Return an error DataFrame if model loading fails
        return pd.DataFrame({"Error": ["Semantic Model failed to load. Check console/internet connection."]})

    city = city.strip().title()
    companies_list = CONCEPTUAL_COMPANIES.get(city, [])
    
    if not companies_list:
        return pd.DataFrame()

    df = pd.DataFrame(companies_list)
    
    # Text to be embedded for semantic comparison (Focus + Status)
    df['embed_text'] = df.apply(lambda row: f"{row['focus']} | Job Status: {row['status']}", axis=1)

    # 1. Generate Target Profile Embedding
    # Note: .cpu() ensures compatibility if the model tries to use a GPU it doesn't have access to
    target_embedding = model.encode([profile_text], convert_to_tensor=True).cpu()
    
    # 2. Generate Company Embeddings
    company_embeddings = model.encode(df['embed_text'].tolist(), convert_to_tensor=True).cpu()

    # 3. Calculate Semantic Score (Cosine Similarity)
    # Cosine similarity requires numpy arrays for scikit-learn
    similarity_matrix = cosine_similarity(target_embedding.numpy(), company_embeddings.numpy())
    
    # Extract scores and assign to the DataFrame
    df['Relevance Score'] = similarity_matrix[0]
    
    # 4. Final Formatting and Ranking
    df['Relevance Score'] = (df['Relevance Score'] * 100).round(2).astype(str) + '%'
    
    df['Active Openings'] = df['status'].apply(lambda s: "✅ YES" if "Hiring" in s else "❌ NO")
    
    df = df.rename(columns={'name': 'Company Name', 'focus': 'Primary Focus', 'status': 'Job Status', 'link': 'Website Link'})
    
    # Rank by relevance (descending)
    df = df.sort_values(by='Relevance Score', ascending=False)
    
    return df[['Company Name', 'Relevance Score', 'Active Openings', 'Job Status', 'Primary Focus', 'Website Link']]

# Example: [No example run here to avoid unnecessary load on startup]