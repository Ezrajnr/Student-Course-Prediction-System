import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Course Prediction Portal", layout="wide")
st.title("🎓 Student Course of Study Prediction System")
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

# Academic Course Tiers & Specializations
ENGINEERING_COURSES = ['Civil Engineering', 'Electrical Engineering', 'Mechanical Engineering']
MEDICAL_COURSES = ['Medicine', 'Pharmacy', 'Nursing', 'Dentistry']
OTHER_PROFESSIONAL_COURSES = ['Law']

PROFESSIONAL_COURSES = ENGINEERING_COURSES + MEDICAL_COURSES + OTHER_PROFESSIONAL_COURSES

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

# --- DYNAMIC UTME SUBJECT SELECTION DROPDOWNS ---
st.subheader("2. UTME Subject Combination & Scores (0–100)")
all_utme_subjects = [
    'English', 'Mathematics', 'Physics', 'Chemistry', 'Biology',
    'Economics', 'Geography', 'Government', 'Commerce', 'Literature'
]

utme_cols = st.columns(5)
utme_selected_subjects = []
utme_scores = []

# Fixed English as 1st Subject, remaining 4 selectable
default_selection = ['English', 'Mathematics', 'Physics', 'Chemistry', 'Biology']

for idx in range(5):
    with utme_cols[idx]:
        st.markdown(f"**Subject {idx+1}**")
        subj = st.selectbox(
            f"Select Subject {idx+1}",
            all_utme_subjects,
            index=all_utme_subjects.index(default_selection[idx]),
            key=f"utme_subj_{idx}"
        )
        score = st.number_input(
            f"Score", 
            min_value=0, 
            max_value=100, 
            value=65 if idx > 0 else 70, 
            key=f"utme_score_{idx}"
        )
        utme_selected_subjects.append(subj)
        utme_scores.append(score)

# Automatic UTME Aggregate Calculation (0–500)
utme_aggregate = sum(utme_scores)
st.info(f"**Calculated UTME Aggregate Score:** {utme_aggregate} / 500")

st.subheader("3. Post-UTME Score (0–100)")
post_utme = st.number_input("Post-UTME Score", 0, 100, 65)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Optimal Course"):
    encoded_olevels = [grade_map[olevel_inputs[s]] for s in olevel_subjects]
    
    # --- ACADEMIC PROFILE ANALYSIS ---
    distinction_count = sum(1 for g in olevel_inputs.values() if g in ['A1', 'B2', 'B3'])
    credit_count = sum(1 for g in olevel_inputs.values() if g in ['C4', 'C5', 'C6'])
    
    is_parallel_distinction = (distinction_count >= 5) and (credit_count == 0)
    is_parallel_credit = (credit_count >= 5) and (distinction_count == 0)
    is_mixed_profile = (distinction_count > 0) and (credit_count > 0)

    # Check for specific distinctions
    has_eng_dist = olevel_inputs['English'] in ['A1', 'B2', 'B3']
    has_math_dist = olevel_inputs['Mathematics'] in ['A1', 'B2', 'B3']
    has_phy_dist = olevel_inputs['Physics'] in ['A1', 'B2', 'B3']
    has_fmath_dist = olevel_inputs['Further_Maths'] in ['A1', 'B2', 'B3']
    has_chem_dist = olevel_inputs['Chemistry'] in ['A1', 'B2', 'B3']
    has_bio_dist = olevel_inputs['Biology'] in ['A1', 'B2', 'B3']

    # Specific track triggers
    has_engineering_distinctions = has_eng_dist and has_math_dist and has_phy_dist and has_fmath_dist
    has_medical_distinctions = has_eng_dist and has_chem_dist and has_bio_dist

    # Social Science Subject Analysis
    social_science_subjs = ['Economics', 'Geography', 'Government', 'Commerce']
    social_in_olevel = any(olevel_inputs[s] in ['A1', 'B2', 'B3', 'C4', 'C5', 'C6'] for s in ['Economics', 'Geography'])
    social_in_utme = any(s in social_science_subjs for s in utme_selected_subjects)
    has_social_bias = social_in_olevel or social_in_utme

    # Core science prerequisite grades
    chem_grade = grade_map[olevel_inputs['Chemistry']]
    bio_grade = grade_map[olevel_inputs['Biology']]
    phy_grade = grade_map[olevel_inputs['Physics']]

    # Model feature vector assembly
    feature_vector = np.array(encoded_olevels + [utme_aggregate, post_utme]).reshape(1, -1)
    
    # Obtain raw model probabilities
    raw_probs = clf_model.predict_proba(feature_vector)[0]
    classes = le.classes_
    course_prob_map = {classes[i]: raw_probs[i] for i in range(len(classes))}

    # --- RULE-BASED HEURISTIC WEIGHT ADJUSTMENTS ---
    adjusted_scores = {}

    for course, prob in course_prob_map.items():
        score = prob

        # CONDITION 1: Specific Distinction Rules (Engineering vs. Medical)
        if has_engineering_distinctions and utme_aggregate >= 250 and post_utme >= 60:
            if course in ENGINEERING_COURSES:
                score *= 8.0
            elif course in PROFESSIONAL_COURSES:
                score *= 1.5

        elif has_medical_distinctions and utme_aggregate >= 250 and post_utme >= 60:
            if course in MEDICAL_COURSES:
                score *= 8.0
            elif course in PROFESSIONAL_COURSES:
                score *= 1.5

        # CONDITION 2: Parallel Credit Tier
        # Criteria: Parallel credits (>=5), UTME 180-200, Post-UTME 40-65
        elif (is_parallel_credit or credit_count >= 5) and (180 <= utme_aggregate <= 200) and (40 <= post_utme <= 65):
            if course in PROFESSIONAL_COURSES:
                score *= 0.0  # Completely block professional courses
            elif has_social_bias:
                if course in SOCIAL_SCIENCE_COURSES:
                    score *= 6.0  # Strongly boost Social Sciences
                elif course in PURE_SCIENCE_COURSES:
                    score *= 0.2
            else:
                if course in PURE_SCIENCE_COURSES:
                    score *= 3.0

        # CONDITION 3: General High-Performance Professional Tier
        elif (is_parallel_distinction or distinction_count >= 5) and utme_aggregate >= 250 and post_utme >= 80:
            if course in PROFESSIONAL_COURSES:
                score *= 4.0

        # CONDITION 4: Mid-Range Pure Science & Social Science Tier
        elif (180 <= utme_aggregate <= 249) and (60 <= post_utme <= 75):
            if course in PROFESSIONAL_COURSES:
                score *= 0.05
            elif has_social_bias and course in SOCIAL_SCIENCE_COURSES:
                score *= 4.0
            elif not has_social_bias and course in PURE_SCIENCE_COURSES:
                score *= 4.0

        # Subject Prerequisite Deficit Penalty
        if chem_grade > 6 or bio_grade > 6 or phy_grade > 6:
            if course in ['Medicine', 'Pharmacy', 'Biochemistry', 'Nursing', 'Dentistry']:
                score *= 0.001

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
    
    # Normalize probabilities for same-tier alternative courses
    alt_total = sum(prob for _, prob in same_tier_matches)
    normalized_alts = []
    if alt_total > 0:
        for course, prob in same_tier_matches:
            scaled_pct = (prob / alt_total) * (100.0 - primary_confidence)
            normalized_alts.append((course, max(scaled_pct, 1.0)))
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
        f"Evaluated on O'Level distinctions (**{strong_str}**), UTME subject combination ({', '.join(utme_selected_subjects)}), "
        f"UTME aggregate score of **{utme_aggregate}/500**, and Post-UTME score of **{post_utme}/100**."
    )
    
    # --- DISPLAY 3 SAME-TIER ALTERNATIVE COURSES ---
    st.markdown(f"### ALTERNATIVE COURSES ({tier_label.upper()} TIER)")
    
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
