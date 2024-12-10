import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import re
from PyPDF2 import PdfReader

# Load SpaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Preprocessing functions
def preprocess_text(text):
    """Clean and preprocess text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.lower().strip()

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyPDF2."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_file(file):
    """Read file content with robust decoding."""
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file.read().decode("ISO-8859-1")
        except UnicodeDecodeError:
            return file.read()  # As raw bytes if decoding fails

def extract_candidate_details(text):
    """Extract candidate details like name, email, and contact."""
    details = {
        'name': "Unknown",
        'email': None,
        'contact': None
    }

    # Extract email
    email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    if email_match:
        details['email'] = email_match.group()

    # Extract phone number
    contact_match = re.search(r'\b\d{10}\b|\+\d{1,3}[-.\s]?\d{9,12}', text)
    if contact_match:
        details['contact'] = contact_match.group()

    # Extract name (simple heuristic: assume the first line is the name)
    lines = text.split("\n")
    for line in lines:
        if re.match(r'^[A-Za-z]+\s[A-Za-z]+$', line.strip()):
            details['name'] = line.strip()
            break

    return details

def match_profiles(job_description, candidate_profiles):
    """Compare job description with candidate profiles."""
    # Filter out empty profiles
    valid_profiles = [profile for profile in candidate_profiles if len(profile['text'].strip()) > 0]

    if len(valid_profiles) == 0:
        raise ValueError("All candidate profiles are empty after preprocessing.")

    # TF-IDF Vectorizer with adjusted parameters
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    job_vector = vectorizer.fit_transform([job_description])
    profile_texts = [profile['text'] for profile in valid_profiles]
    profile_vectors = vectorizer.transform(profile_texts)
    scores = cosine_similarity(job_vector, profile_vectors)
    return scores[0], valid_profiles

# Streamlit UI
st.title("CV Matcher Application")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Home", "Analysis"])

if page == "Home":
    st.header("Home: CV Matching")
    
    # Upload job description
    job_description = st.text_area("Enter Job Description:")
    uploaded_cvs = st.file_uploader("Upload CVs (PDF, Images, or Text):", accept_multiple_files=True)

    if st.button("Analyze"):
        if job_description and uploaded_cvs:
            # Preprocess job description
            job_description_clean = preprocess_text(job_description)

            # Process CVs
            candidate_profiles = []
            for idx, cv in enumerate(uploaded_cvs):
                try:
                    if cv.type.startswith("image/"):
                        text = extract_text_from_image(cv)
                    elif cv.type == "application/pdf":
                        text = extract_text_from_pdf(cv)
                    else:
                        text = read_file(cv)  # Use robust file reading function
                    text_clean = preprocess_text(text)
                    details = extract_candidate_details(text)
                    candidate_profiles.append({
                        'id': f"Candidate {idx+1}",
                        'text': text_clean,
                        'file': cv,
                        'details': details,
                        'raw': text
                    })
                except Exception as e:
                    st.error(f"Error processing file {cv.name}: {e}")

            # Check for empty job description
            if not job_description_clean.strip():
                st.error("The job description is empty after preprocessing.")
            else:
                try:
                    # Match profiles
                    scores, valid_profiles = match_profiles(job_description_clean, candidate_profiles)
                    top_candidates = sorted(
                        enumerate(valid_profiles),
                        key=lambda x: scores[x[0]],
                        reverse=True
                    )[:5]

                    # Display results
                    st.subheader("Top Candidates")
                    for idx, (candidate_idx, profile) in enumerate(top_candidates):
                        details = profile['details']
                        st.write(f"**{profile['id']} ({scores[candidate_idx] * 100:.2f}%)**")
                        st.write(f"- **Name**: {details['name']}")
                        st.write(f"- **Email**: {details['email'] or 'Not Available'}")
                        st.write(f"- **Contact**: {details['contact'] or 'Not Available'}")
                        
                        # CV preview
                        with st.expander(f"Preview {profile['id']} CV"):
                            st.text_area("CV Preview", profile['raw'], height=300)

                        # File download
                        st.download_button(
                            label="Download CV",
                            data=profile['file'].getvalue(),
                            file_name=profile['file'].name,
                            mime=profile['file'].type
                        )
                        st.write("---")

                except ValueError as e:
                    st.error(str(e))
        else:
            st.error("Please provide a job description and upload CVs.")

elif page == "Analysis":
    st.header("Analysis: Candidate Profiles and Insights")
    
    # Mock data for analysis
    all_skills = ["Python", "Data Analysis", "Machine Learning", "Communication", "Leadership"]
    skills_count = {"Python": 5, "Data Analysis": 3, "Machine Learning": 4, "Communication": 6, "Leadership": 2}

    st.subheader("Skills Distribution")
    skill_df = pd.DataFrame(list(skills_count.items()), columns=["Skill", "Count"])
    st.bar_chart(skill_df.set_index("Skill"))

    st.subheader("Top Skills")
    st.write(skill_df.sort_values(by="Count", ascending=False).reset_index(drop=True))

    st.subheader("Candidate Comparisons")
    # Mock comparison data
    candidate_data = pd.DataFrame({
        "Candidate": ["Candidate 1", "Candidate 2", "Candidate 3"],
        "Relevance Score": [95, 89, 87],
        "Years of Experience": [5, 3, 4],
        "Skills Matched": [7, 5, 6]
    })
    st.write(candidate_data)
    
    st.subheader("Relevance Score Distribution")
    st.bar_chart(candidate_data.set_index("Candidate")["Relevance Score"])