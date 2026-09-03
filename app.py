import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- 1. SET PAGE CONFIG & TITLE ---
st.set_page_config(page_title="Student Course Prediction Portal", layout="wide")
st.title("🎓 Student Course Recommendation & Prediction Portal")
st.write("Input the candidate's O'Level grades, UTME subject breakdown, and Post-UTME score to predict course suitability.")

# --- 2. LOAD TRAINED MODEL AND ENCODER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ml_assets():
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    
    # Verify files exist before loading
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error("Model assets ('model.pkl' or 'label_encoder.pkl') were not found in the root directory.")
        st.stop()
        
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

clf_model, le = load_ml_assets()

# Numerical mapping for O'Level grades
grade_map = {'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9}

# Categorize courses by faculty tier
PROFESSIONAL_COURSES = [
    'Medicine', 'Pharmacy', 'Nursing', 'Law', 'Civil Engineering', 
    'Electrical Engineering', 'Mechanical Engineering', 'Dentistry'
]

PURE_SCIENCE_COURSES = [
    'Computer Science', 'Biochemistry', 'Microbiology', 'Industrial Chemistry', 
    'Physics', 'Mathematics', 'Geology', 'Statistics', 'Software Engineering'
]

SOCIAL_SCIENCE_COURSES = [
    'Economics', 'Accounting', 'Business Administration', 
    'Political Science', 'Mass Communication', 'Geography'
]

# --- 3. INPUT FORM UI ---
st.subheader("1. O'Level Subject Grades")
olevel_subjects = [
    'English', 'Mathematics', 'Physics', 'Chemistry', 'Biology', 
    'Geography', 'Agriculture', 'Economics', 'Further_Maths', 'Computer'
]
olevel_inputs = {}

cols1 = st.columns(5)
cols2 = st.columns(5)

for idx, subj in enumerate(olevel_subjects):
    col = cols1[idx] if idx < 5 else cols2[idx - 5]
    with col:
        olevel_inputs[subj] = st.selectbox(
            f"{subj}", 
            ['A1', 'B2', 'B3', 'C4', 'C5', 'C6', 'D7', 'E8', 'F9'],
            index=2 if subj in ['English', 'Mathematics', 'Chemistry'] else 3,
            key=f"ol_{subj}"
        )

st.subheader("2. UTME Subject Scores (0–100)")
utme_cols = st.columns(5)
utme_eng = utme_cols[0].number_input("UTME English", 0, 100, 70)
utme_math = utme_cols[1].number_input("UTME Maths", 0, 100, 65)
utme_phy = utme_cols[2].number_input("UTME Physics", 0, 100, 60)
utme_chem = utme_cols[3].number_input("UTME Chemistry", 0, 100, 58)
utme_bio = utme_cols[4].number_input("UTME Biology", 0, 100, 55)

# Calculate UTME Aggregate automatically (0-500)
utme_aggregate = utme_eng + utme_math + utme_phy + utme_chem + utme_bio
st.info(f"**Calculated UTME Aggregate Score:** {utme_aggregate} / 500")

st.subheader("3. Post-UTME Score (0–100)")
post_utme = st.number_input("Post-UTME Score", 0, 100, 65)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Optimal Course"):
    # Convert O'Level grades to mapped numeric ranks
    encoded_olevels = [grade_map[olevel_inputs[s]] for s in olevel_subjects]
    
    # 1. Analyze O'Level grade quality profile
    distinction_count = sum(1 for g in olevel_inputs.values() if g in ['A1', 'B2', 'B3'])
    credit_count = sum(1 for g in olevel_inputs.values() if g in ['C4', 'C5', 'C6'])
    pass_fail_count = sum(1 for g in olevel_inputs.values() if g in ['D7', 'E8', 'F9'])
    
    # Core requirements check
    math_grade = grade_map[olevel_inputs['Mathematics']]
    eng_grade = grade_map[olevel_inputs['English']]
    chem_grade = grade_map[olevel_inputs['Chemistry']]
    bio_grade = grade_map[olevel_inputs['Biology']]
    phy_grade = grade_map[olevel_inputs['Physics']]

    # Construct complete feature vector for the model
    feature_vector = np.array(encoded_olevels + [utme_aggregate, post_utme]).reshape(1, -1)
    
    # Get raw probabilities from Random Forest model
    raw_probs = clf_model.predict_proba(feature_vector)[0]
    classes = le.classes_
    
    # Create dictionary of courses and raw model probabilities
    course_prob_map = {classes[i]: raw_probs[i] for i in range(len(classes))}

    # --- 2. HEURISTIC ADJUSTMENTS BASED ON ACADEMIC TIER ---
    adjusted_scores = {}
    
    for course, prob in course_prob_map.items():
        score = prob
        
        # Scenario A: Candidate has high Distinctions (A1, B2, B3) & Strong Aggregate
        if distinction_count >= 5 and utme_aggregate >= 250 and post_utme >= 60:
            if course in PROFESSIONAL_COURSES:
                score *= 2.5  # Boost professional course priority
            elif course in PURE_SCIENCE_COURSES:
                score *= 1.2
                
        # Scenario B: Mixed Distinctions & Credits (C4, C5, C6) or Moderate Aggregate
        elif credit_count >= 3 or (distinction_count < 5 and utme_aggregate < 250):
            if course in PROFESSIONAL_COURSES:
                score *= 0.1  # Penalize high-cutoff professional courses
            elif course in PURE_SCIENCE_COURSES:
                score *= 2.0  # Promote pure sciences (Computer Science, Biochemistry, etc.)
            elif course in SOCIAL_SCIENCE_COURSES:
                score *= 1.8  # Promote social sciences
                
        # Scenario C: Deficit in Core Science O'Levels (Chemistry, Biology, Physics > C6)
        if chem_grade > 6 or bio_grade > 6 or phy_grade > 6:
            if course in ['Medicine', 'Pharmacy', 'Biochemistry']:
                score *= 0.05
                
        adjusted_scores[course] = score

    # Normalize adjusted scores to percentages summing to 100%
    total_score = sum(adjusted_scores.values())
    if total_score > 0:
        final_probs = {c: (s / total_score) * 100 for c, s in adjusted_scores.items()}
    else:
        final_probs = {c: 100.0 / len(adjusted_scores) for c in adjusted_scores.keys()}

    # Sort candidates by final probability match
    sorted_courses = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    
    primary_course, primary_confidence = sorted_courses[0]

    # --- 3. DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("RECOMMENDATION")
    
    st.success(f"### Recommended Course: **{primary_course}**\n**Confidence Match:** {primary_confidence:.1f}%")
    
    # Generate dynamic explanation text based on academic tier
    strong_subjects = [subj for subj, grade in olevel_inputs.items() if grade in ['A1', 'B2', 'B3']]
    strong_str = ", ".join(strong_subjects[:3]) if strong_subjects else "core O'Level performance"
    
    if primary_course in PROFESSIONAL_COURSES:
        tier_reason = "exceptional academic record featuring high Distinctions across key prerequisites."
    elif primary_course in PURE_SCIENCE_COURSES:
        tier_reason = "solid balance of Distinction and Credit grades in core science subjects."
    else:
        tier_reason = "balanced credit distribution across general and social science subjects."

    st.caption(
        f"**Reasoning:** Recommended due to a {tier_reason} Strong subject alignment in "
        f"**{strong_str}** combined with a UTME Aggregate of **{utme_aggregate}/500** and Post-UTME score of **{post_utme}/100**."
    )
    
    # --- 4. ALTERNATIVE COURSES DISPLAY ---
    st.markdown("### ALTERNATIVE COURSES")
    alt_cols = st.columns(3)
    
    for i in range(1, 4):
        if i < len(sorted_courses):
            alt_course, alt_conf = sorted_courses[i]
            with alt_cols[i - 1]:
                st.metric(
                    label=f"Option {i}", 
                    value=alt_course, 
                    delta=f"{alt_conf:.1f}% Match"
                )
