# 1. Install required Python packages
!pip install -q streamlit scikit-learn pandas numpy

# 2. Programmatically rewrite app.py with fixed string escaping
app_code = """import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Configure wide layout for modern dashboard feel
st.set_page_config(
    page_title="Student Course of Study Predictor", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished typography, spacing, and dashboard card styling
st.markdown(\"\"\"
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 25px;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: 0.05em;
    }
    .predicted-course {
        font-size: 2.2rem;
        font-weight: 800;
        color: #d97706;
        margin-top: 5px;
    }
    .confidence-score {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 5px;
    }
    .info-banner {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 15px;
    }
    </style>
\"\"\", unsafe_allow_html=True)

@st.cache_resource
def get_trained_model():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE student_records (
        student_id INTEGER PRIMARY KEY,
        utme_score INTEGER,
        post_utme_score INTEGER,
        core_grade INTEGER,
        target_course TEXT
    )''')

    np.random.seed(42)
    samples = 1000
    utme = np.random.randint(180, 360, samples)
    post_utme = np.random.randint(30, 95, samples)
    core_grades = np.random.randint(1, 7, samples) # 1=A1, 2=B2, 3=B3, 4=C4, 5=C5, 6=C6

    courses = []
    for u, p, g in zip(utme, post_utme, core_grades):
        if p >= 75 and u >= 280 and g <= 2:
            courses.append('Medicine')
        elif p >= 70 and u >= 260 and g <= 2:
            courses.append('Electrical Engineering')
        elif p >= 65 and u >= 240 and g <= 3:
            courses.append('Computer Science')
        elif p >= 60 and u >= 230 and g <= 3:
            courses.append('Mechanical Engineering')
        elif p >= 55 and u >= 210 and g <= 4:
            courses.append('Law')
        elif p >= 45 and u >= 190 and g <= 5:
            courses.append('Mass Communication')
        else:
            courses.append('Biochemistry')

    data = list(zip(utme.tolist(), post_utme.tolist(), core_grades.tolist(), courses))
    cursor.executemany('''
        INSERT INTO student_records (utme_score, post_utme_score, core_grade, target_course)
        VALUES (?, ?, ?, ?)
    ''', data)

    df = pd.read_sql_query("SELECT * FROM student_records", conn)
    X = df[['utme_score', 'post_utme_score', 'core_grade']]
    y = df['target_course']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

clf_model = get_trained_model()

# ---------------- SIDEBAR: DASHBOARD SYSTEM CONTROLS ----------------
st.sidebar.markdown("### 🎓 Admin Dashboard")
st.sidebar.markdown("---")
st.sidebar.radio("Navigation", ["Prediction Portal", "Analytics Overview", "System Settings"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Model Status: Ready | Algorithm: Random Forest | Database: SQLite Active")

# ---------------- MAIN PANEL: EXACT SCREENSHOT LAYOUT ----------------

# Header & Subtitle
st.markdown('<div class="main-title">🎓 Student Course of Study Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter student scores to predict the course of study and view confidence.</div>', unsafe_allow_html=True)

# Sliders Row (Post-UTME & UTME)
col1, col2 = st.columns(2)
with col1:
    post_utme_input = st.slider("Post-UTME Score", min_value=0, max_value=100, value=73)
with col2:
    utme_input = st.slider("UTME Score", min_value=180, max_value=400, value=260)

# Selectbox Row (Primary Core Subject Grade)
grade_options = {'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6}
grade_selection = st.selectbox("Primary Core Subject Grade", options=list(grade_options.keys()), index=1) # B2 default
selected_grade_score = grade_options[grade_selection]

st.markdown("---")

# Calculate Button
calculate_btn = st.button("Calculate Prediction")

# Prediction / Result Display
if calculate_btn:
    input_features = pd.DataFrame([[utme_input, post_utme_input, selected_grade_score]],
                                columns=['utme_score', 'post_utme_score', 'core_grade'])

    prediction = clf_model.predict(input_features)[0]
    probabilities = clf_model.predict_proba(input_features)[0]

    class_probs = dict(zip(clf_model.classes_, probabilities))
    confidence_score = class_probs[prediction] * 100

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="metric-label">PREDICTED COURSE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="predicted-course">{prediction}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-label">CONFIDENCE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-score">{confidence_score:.0f}%</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-banner">👈 Enter scores above and click \\'Calculate Prediction\\'.</div>', unsafe_allow_html=True)
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ Fixed app.py written successfully!")
