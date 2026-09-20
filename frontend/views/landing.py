import streamlit as st


def render():
    
    # Landing page CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #9333EA 100%);
            color: white;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(79, 70, 229, 0.3);
        }
        .main-header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ATS Resume Scorer</h1>
        <h3>Optimize Your Resume for Applicant Tracking Systems</h3>
        <p>Get instant feedback on your resume's ATS compatibility with AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call-to-Action Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Analyzing Your Resume", use_container_width=True, type="primary"):
            st.session_state.current_view = 'scorer'
            st.rerun()
    
    st.markdown("---")
    
    # Features Overview
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Comprehensive Scoring
        Get detailed scores across 5 key dimensions (points out of 100):
        - Formatting (20)
        - Keywords & Skills (25)
        - Content Quality (25)
        - Skill Validation (15)
        - ATS Compatibility (15)
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Skill Validation
        Verify that your claimed skills are demonstrated in your projects and experience using AI-powered semantic analysis.
        
        **No more empty claims!**
        """)
    
    with col3:
        st.markdown("""
        ### 🔒 Your Data
        Your resume text is sent to an AI provider (Groq) to extract skills and experience, and the
        results are saved to your account so you can revisit them. The uploaded file itself is not
        stored. You can delete any saved analysis from the History page.
        """)
    
    st.markdown("---")
    
    # How It Works
    st.markdown("## 🚀 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Upload Your Resume
        Supports PDF and DOCX (max 5 MB)
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ AI Analysis
        An AI model plus NLP checks analyze your resume across multiple dimensions
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Get Actionable Feedback
        Receive detailed recommendations to improve your resume
        """)
