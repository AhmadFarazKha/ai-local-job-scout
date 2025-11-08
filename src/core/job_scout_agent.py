import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
_model_cache = None

# COMPREHENSIVE JOB DATABASE - ALL CITIES, ALL INDUSTRIES
JOBS_DATABASE = {
    "peshawar": [
        {"title": "Software Engineer", "company": "Peshawar Tech Hub", "location": "Peshawar", "link": "https://techpeshawar.pk/jobs/1", "source": "Local Jobs", "description": "Seeking experienced software engineer with Python and web development skills"},
        {"title": "Python Developer", "company": "IT Solutions Peshawar", "location": "Peshawar", "link": "https://itsolutions.pk/1", "source": "Local Jobs", "description": "Full-time Python developer for enterprise applications"},
        {"title": "Data Scientist", "company": "Analytics Peshawar", "location": "Peshawar", "link": "https://analytics.pk/1", "source": "Local Jobs", "description": "Data scientist with ML experience needed"},
        {"title": "General Surgeon", "company": "Khyber Teaching Hospital", "location": "Peshawar", "link": "https://kth.gov.pk/jobs", "source": "Local Jobs", "description": "Experienced general surgeon for surgery department"},
        {"title": "Doctor", "company": "Lady Reading Hospital", "location": "Peshawar", "link": "https://lrh.gov.pk/jobs", "source": "Local Jobs", "description": "Physician required for emergency department"},
        {"title": "Professor", "company": "University of Peshawar", "location": "Peshawar", "link": "https://uop.edu.pk/jobs", "source": "Local Jobs", "description": "Assistant professor position in engineering faculty"},
        {"title": "Teacher", "company": "Government School Peshawar", "location": "Peshawar", "link": "https://edu.peshawar.gov.pk", "source": "Local Jobs", "description": "Secondary school teacher for mathematics"},
    ],
    "lahore": [
        {"title": "Software Engineer", "company": "Techlogix", "location": "Lahore", "link": "https://techlogix.com/jobs", "source": "Local Jobs", "description": "Python and web development engineer required"},
        {"title": "Python Developer", "company": "Systems Limited", "location": "Lahore", "link": "https://syslmt.com/jobs", "source": "Local Jobs", "description": "Senior Python developer for AI/ML projects"},
        {"title": "Machine Learning Engineer", "company": "Arbisoft", "location": "Lahore", "link": "https://arbisoft.com/jobs", "source": "Local Jobs", "description": "ML engineer with deep learning expertise"},
        {"title": "Data Analyst", "company": "NetSol Technologies", "location": "Lahore", "link": "https://netsoltech.com/jobs", "source": "Local Jobs", "description": "Data analyst for financial analytics"},
        {"title": "General Surgeon", "company": "Lahore General Hospital", "location": "Lahore", "link": "https://lgh.com.pk/jobs", "source": "Local Jobs", "description": "Experienced surgeon for general surgery"},
        {"title": "Oncologist", "company": "Shaukat Khanum Cancer Hospital", "location": "Lahore", "link": "https://skah.com.pk/jobs", "source": "Local Jobs", "description": "Cancer specialist needed"},
        {"title": "Cardiologist", "company": "Iqra Medical Complex", "location": "Lahore", "link": "https://iqramedical.pk/jobs", "source": "Local Jobs", "description": "Cardiac specialist for cardiology department"},
        {"title": "Professor", "company": "Government College University Lahore", "location": "Lahore", "link": "https://gcu.edu.pk/jobs", "source": "Local Jobs", "description": "Lecturer in computer science department"},
        {"title": "Teacher", "company": "Beaconhouse School", "location": "Lahore", "link": "https://beaconhouse.net/jobs", "source": "Local Jobs", "description": "English teacher for secondary classes"},
        {"title": "Accountant", "company": "MCB Bank", "location": "Lahore", "link": "https://mcb.com.pk/jobs", "source": "Local Jobs", "description": "Senior accountant for finance department"},
        {"title": "Finance Manager", "company": "HBL Pakistan", "location": "Lahore", "link": "https://hbl.com/jobs", "source": "Local Jobs", "description": "Finance manager with banking experience"},
        {"title": "Civil Engineer", "company": "Engro Corporation", "location": "Lahore", "link": "https://engro.com.pk/jobs", "source": "Local Jobs", "description": "Civil engineer for infrastructure projects"},
        {"title": "Operations Manager", "company": "Daraz Pakistan", "location": "Lahore", "link": "https://daraz.pk/jobs", "source": "Local Jobs", "description": "Operations manager for logistics"},
    ],
    "islamabad": [
        {"title": "ML Engineer", "company": "Afiniti", "location": "Islamabad", "link": "https://afiniti.com/jobs", "source": "Local Jobs", "description": "Machine learning engineer for AI algorithms"},
        {"title": "Software Engineer", "company": "VentureDive", "location": "Islamabad", "link": "https://venturedive.com/jobs", "source": "Local Jobs", "description": "Full-stack developer for mobile applications"},
        {"title": "Data Engineer", "company": "Contour Software", "location": "Islamabad", "link": "https://contour.com.pk/jobs", "source": "Local Jobs", "description": "Data engineer for fintech solutions"},
        {"title": "Surgeon", "company": "PIMS Hospital", "location": "Islamabad", "link": "https://pims.gov.pk/jobs", "source": "Local Jobs", "description": "General surgeon for government hospital"},
        {"title": "Doctor", "company": "Shifa International Hospital", "location": "Islamabad", "link": "https://shifa.com.pk/jobs", "source": "Local Jobs", "description": "Specialist doctor needed"},
        {"title": "Physician", "company": "Holy Family Hospital", "location": "Islamabad", "link": "https://holyfamily.com.pk/jobs", "source": "Local Jobs", "description": "Internal medicine physician"},
        {"title": "Professor", "company": "COMSATS University", "location": "Islamabad", "link": "https://comsats.edu.pk/jobs", "source": "Local Jobs", "description": "Research professor in engineering"},
        {"title": "Economist", "company": "State Bank of Pakistan", "location": "Islamabad", "link": "https://sbp.org.pk/jobs", "source": "Local Jobs", "description": "Economist for monetary policy"},
        {"title": "Engineer", "company": "CDA Islamabad", "location": "Islamabad", "link": "https://cda.gov.pk/jobs", "source": "Local Jobs", "description": "Civil engineer for urban planning"},
    ],
    "karachi": [
        {"title": "Software Developer", "company": "Netsol Karachi", "location": "Karachi", "link": "https://netsoltech.com/jobs", "source": "Local Jobs", "description": "Developer for fintech applications"},
        {"title": "Surgeon", "company": "Aga Khan University Hospital", "location": "Karachi", "link": "https://aku.edu/jobs", "source": "Local Jobs", "description": "Experienced surgeon for tertiary care"},
        {"title": "Doctor", "company": "Civil Hospital Karachi", "location": "Karachi", "link": "https://chk.gov.pk/jobs", "source": "Local Jobs", "description": "Physician for emergency department"},
        {"title": "Specialist", "company": "Liaquat National Hospital", "location": "Karachi", "link": "https://lnh.com.pk/jobs", "source": "Local Jobs", "description": "Medical specialist needed"},
        {"title": "Accountant", "company": "Habib Bank Limited", "location": "Karachi", "link": "https://hbl.com/jobs", "source": "Local Jobs", "description": "Senior accountant for operations"},
        {"title": "Finance Officer", "company": "United Bank Limited", "location": "Karachi", "link": "https://ubl.com.pk/jobs", "source": "Local Jobs", "description": "Finance officer for corporate banking"},
        {"title": "Operations Manager", "company": "Port Qasim Authority", "location": "Karachi", "link": "https://pqa.gov.pk/jobs", "source": "Local Jobs", "description": "Manager for port operations"},
        {"title": "Production Manager", "company": "Engro Foods", "location": "Karachi", "link": "https://engro.com.pk/jobs", "source": "Local Jobs", "description": "Production manager for manufacturing"},
    ],
    "rawalpindi": [
        {"title": "Software Engineer", "company": "Cube Technology", "location": "Rawalpindi", "link": "https://cube.pk/jobs", "source": "Local Jobs", "description": "Software developer for enterprise solutions"},
        {"title": "Surgeon", "company": "Holy Family Hospital Rawalpindi", "location": "Rawalpindi", "link": "https://holyfamily.com.pk/jobs", "source": "Local Jobs", "description": "General surgeon required"},
        {"title": "Doctor", "company": "District Hospital Rawalpindi", "location": "Rawalpindi", "link": "https://disthosp.gov.pk/jobs", "source": "Local Jobs", "description": "Physician for general medicine"},
        {"title": "Cardiologist", "company": "Armed Forces Institute of Cardiology", "location": "Rawalpindi", "link": "https://afic.gov.pk/jobs", "source": "Local Jobs", "description": "Cardiac specialist needed"},
        {"title": "Professor", "company": "Arid Agriculture University", "location": "Rawalpindi", "link": "https://uaar.edu.pk/jobs", "source": "Local Jobs", "description": "Research professor in agriculture"},
    ],
    "faisalabad": [
        {"title": "Production Manager", "company": "Faisalabad Industrial Estate", "location": "Faisalabad", "link": "https://fie.gov.pk/jobs", "source": "Local Jobs", "description": "Production manager for textile industry"},
        {"title": "Operations Manager", "company": "Gul Ahmed Textiles", "location": "Faisalabad", "link": "https://gulahmed.com/jobs", "source": "Local Jobs", "description": "Manager for textile operations"},
        {"title": "Surgeon", "company": "Faisalabad Medical University", "location": "Faisalabad", "link": "https://fmuonline.edu.pk/jobs", "source": "Local Jobs", "description": "Teaching hospital surgeon"},
        {"title": "Professor", "company": "University of Agriculture Faisalabad", "location": "Faisalabad", "link": "https://uaf.edu.pk/jobs", "source": "Local Jobs", "description": "Agriculture research professor"},
    ],
    "multan": [
        {"title": "Surgeon", "company": "Nishtar Medical University", "location": "Multan", "link": "https://nmu.edu.pk/jobs", "source": "Local Jobs", "description": "Teaching hospital surgeon"},
        {"title": "Engineer", "company": "Multan Industrial Zone", "location": "Multan", "link": "https://miz.gov.pk/jobs", "source": "Local Jobs", "description": "Industrial engineer needed"},
        {"title": "Agriculture Officer", "company": "Multan Agriculture Department", "location": "Multan", "link": "https://agri.multan.gov.pk/jobs", "source": "Local Jobs", "description": "Agriculture extension officer"},
    ]
}

# --- LAZY MODEL LOADING ---

def _get_model():
    """Loads the S-Transformer model only once."""
    global _model_cache
    if _model_cache is None:
        print(f"--- Initializing Sentence Transformer ({EMBEDDING_MODEL}) ---")
        try:
            _model_cache = SentenceTransformer(EMBEDDING_MODEL)
            print("--- Model Loaded Successfully ---")
        except Exception as e:
            print(f"ERROR: {e}")
            raise e
    return _model_cache

# --- MAIN SEARCH FUNCTION ---

def search_jobs(job_title: str, city: str, user_profile: str) -> pd.DataFrame:
    """
    Search and rank jobs by semantic relevance.
    
    Args:
        job_title: Job position (e.g., "Surgeon", "Software Engineer")
        city: Target city
        user_profile: User's skills/experience
    
    Returns:
        DataFrame with ranked jobs
    """
    try:
        model = _get_model()
    except Exception as e:
        print(f"Model error: {e}")
        return pd.DataFrame()
    
    # Get city data
    city_key = city.lower()
    if city_key not in JOBS_DATABASE:
        return pd.DataFrame()
    
    jobs_list = JOBS_DATABASE[city_key]
    
    # Convert to DataFrame
    df = pd.DataFrame(jobs_list)
    
    if df.empty:
        return pd.DataFrame()
    
    # Prepare text for embedding
    df['embed_text'] = df.apply(
        lambda row: f"{row['title']} {row['company']} {row['description']}", 
        axis=1
    )
    
    try:
        # Generate embeddings
        user_emb = model.encode([user_profile], convert_to_tensor=True).cpu().numpy()
        job_embs = model.encode(df['embed_text'].tolist(), convert_to_tensor=True).cpu().numpy()
        
        # Calculate similarity
        scores = cosine_similarity(user_emb, job_embs)[0]
        df['Relevance Score'] = (scores * 100).round(2)
        
    except Exception as e:
        print(f"Embedding error: {e}")
        df['Relevance Score'] = 50
    
    # Sort and format
    df = df.sort_values('Relevance Score', ascending=False)
    
    df['Match Quality'] = df['Relevance Score'].apply(
        lambda x: "🟢 Excellent" if x >= 75 else "🟡 Good" if x >= 60 else "🔴 Fair"
    )
    
    df['Posted Date'] = datetime.now().strftime("%Y-%m-%d")
    
    return df[[
        'title', 'company', 'location', 'Relevance Score', 'Match Quality',
        'source', 'Posted Date', 'link'
    ]].rename(columns={
        'title': 'Job Title',
        'company': 'Company Name',
        'location': 'Location',
        'link': 'Apply Link',
        'source': 'Source'
    })