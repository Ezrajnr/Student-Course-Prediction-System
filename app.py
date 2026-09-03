import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Or pickle, to load your trained model

# 1. Load your trained Random Forest model and label encoder
# Make sure 'model.pkl' and 'label_encoder.pkl' are in your root directory
@st.cache_resource
def load_ml_assets():
    model = joblib.load('model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, label_encoder

clf_model, le = load_ml_assets()

grade_map = {'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9}

# --- UI INPUT SECTION ---
st.title("Student Course of Study Prediction System")

st.subheader("1. O'Level Subject Grades")
olevel_subjects = ['English', 'Mathematics', 'Physics', 'Chemistry', 'Biology', 
                   'Geography', 'Agriculture', 'Economics', 'Further_Maths', 'Computer']
olevel_inputs = {}

cols1 = st.columns(5)
cols2 = st.columns(5)

for idx, subj in enumerate(olevel_subjects):
    col = cols1[idx] if idx < 5 else cols2[idx - 5]
    with col:
        olevel_inputs[subj] = st.selectbox(subj, ['A1', 'B2', 'B3', 'C4', 'C5', 'C6', 'D7', 'E8', 'F9'])

st.subheader("2. UTME Subject Scores (0-100)")
utme_cols = st.columns(5)
utme_eng = utme_cols[0].number_input("UTME English", 0, 100, 70)
utme_math = utme_cols[1].number_input("UTME Maths", 0, 100, 65)
utme_phy = utme_cols[2].number_input("UTME Physics", 0, 100, 60)
utme_chem = utme_cols[3].number_input("UTME Chemistry", 0, 100, 58)
utme_bio = utme_cols[4].number_input("UTME Biology", 0, 100, 55)

utme_aggregate = utme_eng + utme_math + utme_phy + utme_chem + utme_bio
st.info(f"**Calculated UTME Aggregate Score:** {utme_aggregate} / 500")

st.subheader("3. Post-UTME Score (0-100)")
post_utme = st.number_input("Post-UTME Score", 0, 100, 65)

# --- PREDICTION AND RECOMMENDATION LOGIC ---
if st.button("Predict Optimal Course"):
    # Convert O'Level grades to mapped numeric ranks
    encoded_olevels = [grade_map[olevel_inputs[s]] for s in olevel_subjects]
    
    # Construct complete feature vector matching model training shape
    feature_vector = np.array(encoded_olevels + [utme_aggregate, post_utme]).reshape(1, -1)
    
    # 1. Get class probabilities array from Random Forest
    probabilities = clf_model.predict_proba(feature_vector)[0]
    classes = le.classes_  # List of course names
    
    # 2. Sort probabilities in descending order
    sorted_indices = np.argsort(probabilities)[::-1]
    
    # Primary recommendation (Top 1)
    top_class_idx = sorted_indices[0]
    primary_course = classes[top_class_idx]
    primary_confidence = probabilities[top_class_idx] * 100
    
    # Identify top-performing subject features for dynamic reasoning output
    strong_olevels = [subj for subj, grade in olevel_inputs.items() if grade in ['A1', 'B2', 'B3']]
    strong_utme = []
    if utme_chem >= 50: strong_utme.append("UTME Chemistry")
    if utme_bio >= 50: strong_utme.append("UTME Biology")
    if utme_math >= 50: strong_utme.append("UTME Mathematics")
    if utme_phy >= 50: strong_utme.append("UTME Physics")
    
    olevel_str = ", ".join(strong_olevels[:3]) if strong_olevels else "core subjects"
    utme_str = ", ".join(strong_utme[:2]) if strong_utme else "entrance scores"

    # --- DISPLAY OUTPUT RESULTS ---
    st.markdown("---")
    st.subheader("RECOMMENDATION")
    
    st.success(f"**Course:** {primary_course}\n\n**Confidence:** {primary_confidence:.1f}%")
    st.caption(
        f"Strong match on {olevel_str} (O'Level) and {utme_str} (UTME) "
        f"combined with the aggregate score ({utme_aggregate}/500) and post-UTME performance ({post_utme}/100)."
    )
    
    # 3. Display Top 3 Alternative Courses
    st.subheader("ALTERNATIVE COURSES")
    
    alt_col1, alt_col2, alt_col3 = st.columns(3)
    
    for i, col in enumerate([alt_col1, alt_col2, alt_col3]):
        if len(sorted_indices) > i + 1:
            alt_idx = sorted_indices[i + 1]
            alt_course = classes[alt_idx]
            alt_conf = probabilities[alt_idx] * 100
            
            with col:
                st.metric(label=f"Option {i+1}", value=alt_course, delta=f"{alt_conf:.1f}% Match")
