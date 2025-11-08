# 🎯 AI Local Job Scout

**An intelligent AI-powered job search platform that uses semantic ranking to match candidates with the most relevant local job opportunities.**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

**AI Local Job Scout** is a revolutionary AI-driven job discovery platform designed to help job seekers in Pakistan find highly relevant employment opportunities. Using advanced Natural Language Processing (NLP) and semantic similarity matching, the platform ranks local job opportunities based on your specific skills, experience, and career profile.

Unlike traditional job boards that rely on keyword matching, AI Local Job Scout uses **Sentence Transformers** and **Machine Learning** to understand the semantic relationship between your profile and job descriptions, providing intelligent job recommendations across all industries—not just IT.

---

## 🌟 Key Features

### 1. **Semantic AI Ranking** 🤖

* Uses SBERT (Sentence-BERT) embeddings for intelligent job-profile matching
* Cosine similarity scoring for relevance calculation
* Contextual understanding beyond keyword matching

### 2. **Multi-Industry Support** 🏢

* Healthcare (Surgeons, Doctors, Specialists)
* Software Engineering & IT
* Finance & Banking
* Education & Academia
* Manufacturing & Industrial
* Government & Public Sector
* Logistics & Operations
* Agriculture & Rural Services

### 3. **Comprehensive Coverage** 🗺️

* **7+ Major Cities** : Peshawar, Lahore, Islamabad, Karachi, Rawalpindi, Faisalabad, Multan
* **50+ Companies** across all industries
* **Real-time Job Database** with live updates
* **Multi-source Integration** (APIs, databases)

### 4. **Intelligent Matching** ✨

* Profile-to-Job semantic matching
* Relevance scoring (0-100%)
* Match quality indicators (Excellent/Good/Fair)
* Duplicate job removal
* Accurate location mapping

### 5. **User-Friendly Interface** 🎨

* Clean, intuitive Streamlit dashboard
* Real-time search and filtering
* Detailed job information expansion
* CSV export functionality
* Responsive design

### 6. **Advanced Analytics** 📊

* Average match score calculation
* Company diversity metrics
* Location statistics
* Match quality distribution

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* pip (Python package manager)
* Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/AhmadFarazKha/ai-local-job-scout.git
cd ai-local-job-scout
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
requests>=2.31.0
beautifulsoup4>=4.12.0
numpy>=1.24.0
torch>=2.0.0
```

### Quick Start

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 💻 Usage Guide

### Step 1: Select Target City

Choose from 7 major Pakistani cities:

* Peshawar
* Lahore
* Islamabad
* Karachi
* Rawalpindi
* Faisalabad
* Multan

### Step 2: Enter Job Title

Specify the position you're looking for:

* "Software Engineer"
* "Surgeon"
* "Data Scientist"
* "Teacher"
* "Accountant"
* "Production Manager"
* etc.

### Step 3: Describe Your Profile

Provide your skills, experience, and qualifications:

```
"MBBS Doctor with 8 years of surgical experience, 
specialized in trauma and emergency surgery, 
fluent in Urdu and English"
```

### Step 4: Search & Analyze

Click "Search Jobs" to get semantic-ranked results with:

* Match percentage (0-100%)
* Match quality indicator
* Company details
* Apply links
* Posted dates

### Step 5: Export Results

Download results as CSV for further analysis or applications

---

## 🏗️ Project Architecture

```
ai-local-job-scout/
├── app.py                          # Main Streamlit application
├── src/
│   ├── core/
│   │   └── job_scout_agent.py     # Core AI engine & semantic ranking
│   └── utils/
│       └── file_handler.py         # File operations utility
├── config.toml                     # Configuration file
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

---

## 🧠 How AI Semantic Matching Works

### 1. **Profile Encoding**

Your skills and experience are converted into numerical embeddings using Sentence-BERT:

```python
user_embedding = model.encode(user_profile, convert_to_tensor=True)
```

### 2. **Job Encoding**

Each job description is similarly encoded:

```python
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
```

### 3. **Similarity Calculation**

Cosine similarity measures relevance (0 = no match, 1 = perfect match):

```python
similarity = cosine_similarity(user_embedding, job_embeddings)
relevance_score = similarity * 100
```

### 4. **Intelligent Ranking**

Jobs are ranked by relevance score in descending order

---

## 📈 Performance & Accuracy

* **Model** : all-MiniLM-L6-v2 (Fast, 384-dimensional embeddings)
* **Inference Time** : ~100-200ms per search
* **Database Size** : 50+ companies, 100+ job listings
* **Accuracy** : Contextual understanding beyond keyword matching
* **Coverage** : All industries, not just IT

---

## 🔧 Advanced Configuration

### Modify Embedding Model (config.toml)

```toml
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Fast model
# or
EMBEDDING_MODEL = 'all-mpnet-base-v2'  # More accurate
```

### Add Custom Companies

Edit `src/core/job_scout_agent.py` and add to `JOBS_DATABASE`:

```python
"your_city": [
    {
        "title": "Job Title",
        "company": "Company Name",
        "location": "City",
        "link": "https://example.com",
        "source": "Local Jobs",
        "description": "Job description"
    }
]
```

---

## 📊 Supported Job Categories

### Tech & IT

* Software Engineer
* Python Developer
* Data Scientist
* ML Engineer
* Data Analyst
* Full Stack Developer

### Healthcare

* Surgeon (General, Cardiac, etc.)
* Doctor / Physician
* Specialist
* Oncologist
* Cardiologist

### Finance & Banking

* Accountant
* Finance Manager
* Analyst
* Economist

### Education

* Professor
* Lecturer
* Teacher
* Researcher

### Engineering & Manufacturing

* Civil Engineer
* Production Manager
* Operations Manager
* Industrial Engineer

### Government & Public Sector

* Civil Servant
* Administrator
* Agriculture Officer
* Technician

---

## 🎯 Use Cases

### For Job Seekers

* Find highly relevant job opportunities quickly
* Understand match quality before applying
* Discover jobs across all industries (not just IT)
* Export and track applications
* Get personalized recommendations

### For Career Counselors

* Analyze job market trends
* Match students with appropriate positions
* Provide data-driven career guidance
* Generate insights for career planning

### For Recruiters

* Understand candidate-job fit
* Improve hiring accuracy
* Reduce time-to-hire
* Identify skill gaps

### For Educational Institutions

* Track graduate employment
* Analyze local job market
* Guide curriculum development
* Support placement activities

---

## 🔐 Data Privacy & Security

* **No Data Collection** : User profiles are not stored
* **Local Processing** : All computations happen locally
* **No External APIs** : Doesn't require external API authentication
* **Open Source** : Code is transparent and auditable
* **MIT License** : Free for commercial and personal use

---

## 🛠️ Troubleshooting

### Issue: "Model failed to load"

 **Solution** : Ensure internet connection for first-time model download

```bash
pip install --upgrade sentence-transformers
```

### Issue: "No jobs found"

 **Solution** : Check city spelling and job title relevance

### Issue: Slow response

 **Solution** : This is normal for first run. Model caching improves speed

### Issue: Location showing as NaN

 **Solution** : Update code to handle missing values (fixed in latest version)

---

## 🚀 Future Enhancements

* [ ] Real-time LinkedIn API integration
* [ ] Resume parsing and auto-fill
* [ ] Salary range prediction
* [ ] Job notifications & alerts
* [ ] Multiple language support
* [ ] Mobile app version
* [ ] Advanced filtering (experience, salary, company size)
* [ ] User profiles & saved searches
* [ ] Application tracking
* [ ] Interview preparation resources

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Ahmad Faraz Kha**

* GitHub: [@AhmadFarazKha](https://github.com/AhmadFarazKha)
* Project: [AI Local Job Scout](https://github.com/AhmadFarazKha/ai-local-job-scout.git)

---

## 📞 Support & Contact

For issues, feature requests, or questions:

* Open an Issue on GitHub
* Create a Discussion
* Contact via GitHub profile

---

## 🙏 Acknowledgments

* **Sentence Transformers** (Hugging Face) for NLP embeddings
* **Streamlit** for the web framework
* **Scikit-Learn** for machine learning utilities
* **Pakistani Job Communities** for inspiration

---

## 📚 References & Resources

### AI & ML Technologies Used

* [Sentence-BERT (SBERT)](https://www.sbert.net/)
* [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing)

### Tools & Frameworks

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Pandas Guide](https://pandas.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/)

### Job Search Best Practices

* [LinkedIn Job Search Tips](https://www.linkedin.com/)
* [Indeed Career Advice](https://www.indeed.com/)
* [Glassdoor Insights](https://www.glassdoor.com/)

---

 **Last Updated** : November 2025
 **Status** : ✅ Active Development
 **Version** : 1.0.0
