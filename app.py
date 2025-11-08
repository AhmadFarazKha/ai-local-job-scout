import streamlit as st
import pandas as pd
from src.core.job_scout_agent import search_jobs

# Page config
st.set_page_config(page_title="AI Local Job Scout", page_icon="🎯", layout="wide")

st.title("🎯 AI Local Job Scout")
st.markdown("**Find and rank local job opportunities based on AI semantic matching**")

# Sidebar
with st.sidebar:
    st.header("1. Define Your Target")
    
    city = st.selectbox(
        "Target City",
        ["Peshawar", "Lahore", "Islamabad", "Karachi", "Rawalpindi", "Faisalabad", "Multan"]
    )
    
    job_title = st.text_input(
        "Job Title",
        placeholder="e.g., Surgeon, Software Engineer, Teacher",
        value="Software Engineer"
    )
    
    user_profile = st.text_area(
        "Your Profile / Skills",
        placeholder="Describe your experience and skills",
        height=100,
        value="Software engineer with Python expertise"
    )
    
    search_btn = st.button("🔍 Search Jobs", use_container_width=True)

# Main content
if search_btn:
    if not job_title or not user_profile:
        st.error("❌ Please fill all fields")
    else:
        with st.spinner("⏳ Searching..."):
            results = search_jobs(job_title, city, user_profile)
            
            if results.empty:
                st.warning(f"⚠️ No jobs found for '{job_title}' in {city}")
            else:
                # Stats
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Match Score", f"{results['Relevance Score'].mean():.1f}%")
                col2.metric("🏢 Jobs Found", len(results))
                col3.metric("✅ Excellent Match", len(results[results['Relevance Score'] >= 75]))
                
                st.markdown("---")
                
                # Results table
                st.subheader(f"2. Results for '{job_title}' in {city}")
                
                display_df = results.copy()
                display_df['Relevance Score'] = display_df['Relevance Score'].astype(str) + "%"
                
                st.dataframe(
                    display_df[['Job Title', 'Company Name', 'Location', 'Relevance Score', 'Match Quality', 'Source']],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # Detailed view
                st.subheader("📋 Job Details")
                
                for idx, row in results.iterrows():
                    with st.expander(f"{row['Company Name']} - {row['Job Title']} ({row['Relevance Score']:.1f}%)"):
                        st.write(f"**Location:** {row['Location']}")
                        st.write(f"**Match Quality:** {row['Match Quality']}")
                        st.write(f"**Posted:** {row['Posted Date']}")
                        st.write(f"**Source:** {row['Source']}")
                        if row['Apply Link']:
                            st.markdown(f"**[Apply Now →]({row['Apply Link']})**")
                
                # Download
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name=f"jobs_{city}_{job_title}.csv"
                )
else:
    st.info("👈 Enter your search criteria and click 'Search Jobs'")