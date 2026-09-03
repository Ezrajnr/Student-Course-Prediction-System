import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Course Prediction Portal", layout="wide")
st.title("🎓 Student Course Recommendation & Prediction Portal")
st.write("Input the candidate's O'Level grades, UTME subject breakdown, and Post-UTME score to predict course suitability.")

# --- 2. LOAD MODEL ASSETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ml_assets():
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error("Model assets ('model.pkl' or 'label_encoder.pkl') were not found in the root directory.")
        st.stop()
        
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

clf_model, le = load_ml_assets()

# Numerical mapping for O'Level grades
grade_map = {'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9}

# Define academic course tiers
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

def get_course_tier(course_name):
    if course_name in PROFESSIONAL_COURSES:
        return 'Professional', PROFESSIONAL_COURSES
    elif course_name in PURE_SCIENCE_COURSES:
        return 'Pure Science', PURE_SCIENCE_COURSES
    else:
        return 'Social Science', SOCIAL_SCIENCE_COURSES

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

# Automatic UTME Aggregate Calculation (0–500)
utme_aggregate = utme_eng + utme_math + utme_phy + utme_chem + utme_bio
st.info(f"**Calculated UTME Aggregate Score:** {utme_aggregate} / 500")

st.subheader("3. Post-UTME Score (0–100)")
post_utme = st.number_input("Post-UTME Score", 0, 100, 65)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Optimal Course"):
    encoded_olevels = [grade_map[olevel_inputs[s]] for s in olevel_subjects]
    
    # Analyze O'Level grade quality
    distinction_count = sum(1 for g in olevel_inputs.values() if g in ['A1', 'B2', 'B3'])
    credit_count = sum(1 for g in olevel_inputs.values() if g in ['C4', 'C5', 'C6'])
    
    chem_grade = grade_map[olevel_inputs['Chemistry']]
    bio_grade = grade_map[olevel_inputs['Biology']]
    phy_grade = grade_map[olevel_inputs['Physics']]

    feature_vector = np.array(encoded_olevels + [utme_aggregate, post_utme]).reshape(1, -1)
    
    # Obtain raw model probabilities
    raw_probs = clf_model.predict_proba(feature_vector)[0]
    classes = le.classes_
    course_prob_map = {classes[i]: raw_probs[i] for i in range(len(classes))}

    # --- HEURISTIC WEIGHT ADJUSTMENTS ---
    adjusted_scores = {}
    
    for course, prob in course_prob_map.items():
        score = prob
        
        # Criterion: 5+ Distinctions AND UTME >= 250 AND Post-UTME >= 60
        if distinction_count >= 5 and utme_aggregate >= 250 and post_utme >= 60:
            if course in PROFESSIONAL_COURSES:
                score *= 3.0  # Boost professional course weights
            elif course in PURE_SCIENCE_COURSES:
                score *= 1.1
        # Mixed Credits or lower test aggregates
        elif credit_count >= 3 or (distinction_count < 5 and utme_aggregate < 250):
            if course in PROFESSIONAL_COURSES:
                score *= 0.05  # Strongly reduce professional eligibility
            elif course in PURE_SCIENCE_COURSES:
                score *= 2.0
            elif course in SOCIAL_SCIENCE_COURSES:
                score *= 1.8
                
        # Subject deficit check
        if chem_grade > 6 or bio_grade > 6 or phy_grade > 6:
            if course in ['Medicine', 'Pharmacy', 'Biochemistry']:
                score *= 0.01
                
        adjusted_scores[course] = score

    # Normalize adjusted scores to percentages
    total_score = sum(adjusted_scores.values())
    if total_score > 0:
        final_probs = {c: (s / total_score) * 100 for c, s in adjusted_scores.items()}
    else:
        final_probs = {c: 100.0 / len(adjusted_scores) for c in adjusted_scores.keys()}

    # Rank overall courses by probability
    sorted_all_courses = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Primary recommendation
    primary_course, primary_confidence = sorted_all_courses[0]
    
    # Determine the faculty tier of the primary course
    tier_label, same_tier_courses = get_course_tier(primary_course)
    
    # --- FILTER ALTERNATIVES TO SAME TIER ONLY ---
    same_tier_matches = [
        (course, prob) for course, prob in sorted_all_courses 
        if course in same_tier_courses and course != primary_course
    ]
    
    # Normalize probabilities for the same-tier alternative courses
    alt_total = sum(prob for _, prob in same_tier_matches)
    normalized_alts = []
    if alt_total > 0:
        for course, prob in same_tier_matches:
            # Scale relative to remaining probability mass
            scaled_pct = (prob / alt_total) * (100.0 - primary_confidence)
            normalized_alts.append((course, max(scaled_pct, 5.0)))
    else:
        normalized_alts = same_tier_matches

    # Sort tier alternatives
    normalized_alts = sorted(normalized_alts, key=lambda x: x[1], reverse=True)

    # --- DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("RECOMMENDATION")
    
    st.success(f"### Course: **{primary_course}**\n**Confidence:** {primary_confidence:.1f}%")
    
    # Dynamic reasoning summary
    strong_subjects = [subj for subj, grade in olevel_inputs.items() if grade in ['A1', 'B2', 'B3']]
    strong_str = ", ".join(strong_subjects[:3]) if strong_subjects else "core prerequisites"
    
    st.caption(
        f"**Reasoning:** Recommended under the **{tier_label}** classification. "
        f"Strong match on **{strong_str}** (O'Level) combined with a UTME Aggregate score of **{utme_aggregate}/500** "
        f"and Post-UTME performance of **{post_utme}/100**."
    )
    
    # --- DISPLAY 3 SAME-TIER ALTERNATIVE COURSES ---
    st.markdown(f"### ALTERNATIVE COURSES ({tier_label.upper()} TIERS)")
    
    alt_cols = st.columns(3)
    
    for i in range(3):
        with alt_cols[i]:
            if i < len(normalized_alts):
                alt_course, alt_conf = normalized_alts[i]
                st.metric(
                    label=f"Alternative {i+1}", 
                    value=alt_course, 
                    delta=f"{alt_conf:.1f}% Match"
                )
            else:
                st.info("No further courses available in this category.")
