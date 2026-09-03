import streamlit as st
import numpy as np
import pandas as pd

# 1. O'Level Mapping Dictionary
grade_map = {'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9}

st.title("Admin Student Course Prediction Portal")

# 2. O'Level Inputs
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

# 3. UTME Inputs (0-100)
st.subheader("2. UTME Subject Scores (0-100)")
utme_cols = st.columns(5)
utme_eng = utme_cols[0].number_input("UTME English", 0, 100, 70)
utme_math = utme_cols[1].number_input("UTME Maths", 0, 100, 65)
utme_phy = utme_cols[2].number_input("UTME Physics", 0, 100, 60)
utme_chem = utme_cols[3].number_input("UTME Chemistry", 0, 100, 58)
utme_bio = utme_cols[4].number_input("UTME Biology", 0, 100, 55)

# Automatic UTME Aggregate Calculation (0-500)
utme_aggregate = utme_eng + utme_math + utme_phy + utme_chem + utme_bio
st.info(f"**Calculated UTME Aggregate Score:** {utme_aggregate} / 500")

# 4. Post-UTME Input (0-100)
st.subheader("3. Post-UTME Score (0-100)")
post_utme = st.number_input("Post-UTME Score", 0, 100, 65)

# 5. Prediction Execution
if st.button("Predict Optimal Course"):
    # Convert O'Level grades to mapped numeric ranks
    encoded_olevels = [grade_map[olevel_inputs[s]] for s in olevel_subjects]
    
    # Construct complete feature vector for the ML model
    feature_vector = np.array(encoded_olevels + [utme_aggregate, post_utme]).reshape(1, -1)
    
    # Execute prediction (assuming clf_model is loaded)
    # prediction = clf_model.predict(feature_vector)
    # probability = clf_model.predict_proba(feature_vector)
    
    st.success("Feature vector constructed successfully! Ready for Random Forest classification.")
