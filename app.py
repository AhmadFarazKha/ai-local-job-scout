import streamlit as st
import pandas as pd
from src.core.job_scout_agent import get_tech_industry_data
from src.utils.file_handler import create_directories_if_not_exist
from dotenv import load_dotenv

# Ensure consistency and load environment variables
create_directories_if_not_exist("data")
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Local Job Scout",
    page_icon="🗺️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Initial State Setup ---
if 'city_input' not in st.session_state:
    st.session_state.city_input = "Lahore"
if 'profile_input' not in st.session_state:
    st.session_state.profile_input = "AI Engineer with strong skills in Computer Vision, Python, MLOps, and Data Science."

# --- Application Content ---
st.title("🗺️ AI Local Tech Job Scout")
st.markdown("<p style='text-align: center; color: var(--text-color); font-size: 1.1rem; margin-bottom: 2rem;'>Find and rank local tech companies based on semantic match with your specific AI/ML profile.</p>", unsafe_allow_html=True)

col_main_l, col_main_center, col_main_r = st.columns([1, 4, 1])

with col_main_center:
    with st.container(border=True):
        st.markdown("### 1. Define Your Target")
        
        st.session_state.city_input = st.text_input(
            "Target City:",
            value=st.session_state.city_input,
            placeholder="e.g., Lahore, Islamabad, Mianwali",
            key="city_input_key"
        )
        
        st.session_state.profile_input = st.text_area(
            "Your Profile / Desired Skills:",
            value=st.session_state.profile_input,
            placeholder="e.g., AI Engineer with strong skills in Computer Vision, Python, MLOps, and Data Science.",
            height=150,
            key="profile_input_key"
        )
        
        st.markdown("---")
        
        if st.button("Search & Rank Local Opportunities 🔎", use_container_width=True, key="btn_search"):
            if not st.session_state.city_input or not st.session_state.profile_input:
                st.warning("Please enter both the City and your Profile/Skills.")
                st.stop()
            
            with st.spinner(f"Analyzing {st.session_state.city_input}'s tech market and ranking results..."):
                try:
                    search_results_df = get_tech_industry_data(st.session_state.city_input, st.session_state.profile_input)
                    
                    st.markdown("---")
                    st.subheader(f"2. Ranked Opportunities in {st.session_state.city_input.title()}")

                    if search_results_df.empty:
                         st.warning(f"No specific company data found in our database for {st.session_state.city_input.title()}.")
                    else:
                        st.success("Results ranked by semantic relevance to your profile!")
                        
                        # Display results using markdown table for better link rendering
                        st.markdown(
                            search_results_df.to_html(escape=False), 
                            unsafe_allow_html=True
                        )
                        
                except Exception as e:
                    # Note: First run will download the Sentence Transformer model (100MB+)
                    st.error(f"An unexpected error occurred during search: {e}")
                    st.error("If this is the first run, ensure your internet connection is stable (the AI model needs to download ~100MB).")

st.markdown("---")
st.info("Powered by Semantic Ranking (SentenceTransformers/Scikit-learn) and Python.")
st.markdown("<p style='text-align: center; color: #a0a0a0; margin-top: 2rem;'>Developed with ❤️ in Mianwali, Punjab, Pakistan</p>", unsafe_allow_html=True)