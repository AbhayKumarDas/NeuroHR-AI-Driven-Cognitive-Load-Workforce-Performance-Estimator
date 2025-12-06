import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="NeuroHR", page_icon="🧠", layout="centered")

# ============ TITLE ============
st.title("NeuroHR: AI Driven Cognitive Load and Workforce Performance Estimator")

# ============ PROJECT DESCRIPTION ============
st.markdown(""" **Project Description:** NeuroHR is an AI-powered system that estimates employee **Cognitive Load Index (CLI)** in real-time using behavioral and workload indicators. The project helps HR departments and managers identify burnout risks, predict performance drops when assigning new tasks, and make proactive decisions on workload redistribution.

What is Cognitive Load Index (CLI)?

CLI is a quantitative measure (0–1) of an employee's mental and workload stress level:

- **Low CLI (0.0–0.3)**: Employee is well-balanced and can handle more responsibility
- **Medium CLI (0.3–0.55)**: Employee is approaching moderate workload stress
- **High CLI (0.55–0.75)**: Employee shows early signs of fatigue and performance decline
- **Critical CLI (0.75–1.0)**: Employee is likely experiencing burnout and needs intervention
""")

# ============ TECH STACK ============
st.markdown("""**Tech Stack:** Python, Streamlit, Scikit-learn (Random Forest Regressor, KMeans Clustering, PCA, StandardScaler, OneHotEncoder, ColumnTransformer,Pipeline, Cross-Validation), Pandas, NumPy, Matplotlib, Seaborn, FPDF (PDF Report Generation), Pickle (Model Serialization)""")

st.markdown("---")

# ============ LOAD MODELS ============
@st.cache_resource
def load_models():
    with open('models/cli_model.pkl', 'rb') as f:
        cli_model = pickle.load(f)
    with open('models/delta_model.pkl', 'rb') as f:
        delta_model = pickle.load(f)
    with open('models/clustering_artifacts.pkl', 'rb') as f:
        clustering = pickle.load(f)
    return cli_model, delta_model, clustering

try:
    cli_model, delta_model, clustering = load_models()
    kmeans = clustering['kmeans_model']
    scaler = clustering['scaler']
    cluster_names = clustering['cluster_interpretations']
except:
    st.error("Models not found in models/ folder.")
    st.stop()

# ============ EMPLOYEE DETAILS ============
st.subheader("Employee Details")

col1, col2 = st.columns(2)
with col1:
    employee_name = st.text_input("Name")
    age = st.number_input("Age", 18, 65, 30)
    role = st.selectbox("Role Level", ["Junior", "Mid", "Senior", "Lead", "C-Level"])

with col2:
    department = st.selectbox("Department", ["Engineering", "DataScience", "Product", "Sales", "Finance", "HR", "Operations", "Marketing"])
    project_phase = st.selectbox("Project Phase", ["Planning", "Development", "Testing", "Deployment", "Maintenance"])
    persona = st.selectbox("Employee Persona", ["Balanced", "Resilient", "Sensitive", "Workaholic", "Struggling"])

st.markdown("---")

# ============ PARAMETER TUNING ============
st.subheader("Parameter Tuning")

col3, col4 = st.columns(2)
with col3:
    weekly_hours = st.slider("Weekly Work Hours", 20, 100, 40)
    overtime = st.slider("Overtime (last 30 days)", 0, 120, 10)
    active_tasks = st.slider("Active Tasks", 1, 50, 5)

with col4:
    stress = st.slider("Stress Level (1-5)", 1.0, 5.0, 3.0, 0.5)
    performance = st.slider("Performance Rating (1-5)", 1.0, 5.0, 3.5, 0.5)
    wellbeing = st.slider("Wellbeing Score (0-100)", 0, 100, 70)

st.markdown("---")

# ============ ANALYZE BUTTON ============
if st.button("🔍 Analyze", use_container_width=True):

    if not employee_name.strip():
        st.warning("Please enter employee name.")
        st.stop()

    # Prepare input
    cli_input = pd.DataFrame({
        'Weekly_Workload_Hours': [weekly_hours],
        'Workload_to_Capacity_Ratio': [weekly_hours / 45],
        'Self_Reported_Stress': [stress],
        'Task_Switch_Frequency': [active_tasks * 3],
        'Active_Task_Count': [active_tasks],
        'Overtime_Hours_30d': [overtime],
        'After_Hours_Activity_Hours': [overtime * 0.3],
        'Focus_Time_Ratio': [max(0.2, 1 - (stress / 10))],
        'Sentiment_Score': [0.5 - (stress / 10)],
        'Salary_to_Experience_Ratio': [0.5],
        'Perf_Trend_Slope': [0.0],
        'Performance_Rating_Current': [performance],
        'Avg_Daily_Meeting_Hours': [2.0],
        'Task_Overdue_Count': [max(0, active_tasks - 3)],
        'Task_Completion_Rate': [performance / 5],
        'Avg_Response_Time_Hours': [3.0],
        'Days_Since_Vacation': [60],
        'Leave_Days_3m': [2],
        'Performance_Volatility': [0.2],
        'Skill_Role_Match_Score': [0.8],
        'Capacity_Score': [1.0],
        'Manager_Support_Score': [0.8],
        'Survey_Wellbeing_Score': [wellbeing],
        'Language_Tone_Stress_Index': [max(0, (stress - 1) / 4)],
        'Role_Level': [role],
        'Department': [department],
        'Project_Phase': [project_phase],
        'Employee_Persona': [persona]
    })

    # Predict CLI
    cli = float(np.clip(cli_model.predict(cli_input)[0], 0, 1))

    # Risk category
    if cli < 0.30:
        risk = "Low"
    elif cli < 0.55:
        risk = "Medium"
    elif cli < 0.75:
        risk = "High"
    else:
        risk = "Critical"

    # Predict performance drop
    delta_input = cli_input.copy()
    delta_input['CLI_v2'] = cli
    delta_input['Risk_Category'] = risk
    delta_input['Recent_Performance_Delta'] = 0.0
    delta = float(delta_model.predict(delta_input)[0])

    # Cluster
    cluster_input = np.array([[cli, stress, weekly_hours, weekly_hours/45, active_tasks*3, max(0.2, 1-(stress/10)), overtime, overtime*0.3, 0.5-(stress/10), wellbeing, max(0, (stress-1)/4)]])
    cluster_scaled = scaler.transform(cluster_input)
    cluster_id = int(kmeans.predict(cluster_scaled)[0])
    cluster_name = cluster_names.get(cluster_id, "Unknown")

    # Burnout probability
    burnout_prob = float(1 / (1 + np.exp(-10 * (cli - 0.55))))

    # Calculate capacity based on persona
    persona_capacity = {"Balanced": 1.0, "Resilient": 1.25, "Sensitive": 0.85, "Workaholic": 1.1, "Struggling": 0.7}
    capacity = persona_capacity.get(persona, 1.0)

    # Skill match (derived from performance)
    skill_match = min(1.0, 0.5 + (performance - 3.0) / 4.0)

    # HR Recommendation (matching original notebook logic)
    if cli >= 0.75 or delta <= -1.0 or capacity < 0.7:
        recommendation = "Do Not Assign New Tasks"
        rec_desc = "Critical burnout risk. Reduce workload immediately."
    elif cli >= 0.55 or delta <= -0.6 or skill_match < 0.5:
        recommendation = "Assign With Support"
        rec_desc = "Needs additional support and monitoring."
    elif skill_match > 0.75 and cli < 0.40 and delta > -0.3:
        recommendation = "High Priority - Ready for More"
        rec_desc = "Has capacity for additional work."
    else:
        recommendation = "Assign Normally"
        rec_desc = "Balanced state, can be assigned tasks normally."

    # Store results
    st.session_state['results'] = {
        'name': employee_name,
        'age': age,
        'role': role,
        'department': department,
        'project_phase': project_phase,
        'persona': persona,
        'weekly_hours': weekly_hours,
        'overtime': overtime,
        'stress': stress,
        'performance': performance,
        'wellbeing': wellbeing,
        'active_tasks': active_tasks,
        'cli': cli,
        'risk': risk,
        'burnout_prob': burnout_prob,
        'delta': delta,
        'cluster_id': cluster_id,
        'cluster_name': cluster_name,
        'recommendation': recommendation,
        'rec_desc': rec_desc,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ============ RESULTS ============
if 'results' in st.session_state:
    r = st.session_state['results']

    st.markdown("---")
    st.subheader(f"Results: {r['name']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CLI Score", f"{r['cli']:.2f}")
    m2.metric("Risk Level", r['risk'])
    m3.metric("Burnout Risk", f"{r['burnout_prob']*100:.0f}%")
    m4.metric("Perf Impact", f"{r['delta']:.2f}")

    st.markdown("---")

    if r['risk'] == "Critical":
        st.error(f"**{r['recommendation']}** - {r['rec_desc']}")
    elif r['risk'] == "High":
        st.warning(f"**{r['recommendation']}** - {r['rec_desc']}")
    else:
        st.success(f"**{r['recommendation']}** - {r['rec_desc']}")

    st.info(f"**Employee Profile:** {r['cluster_name']}")

    st.markdown("---")

    # PDF Generation
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pw = pdf.w - 20  # page width minus margins

        # Header with background
        pdf.set_fill_color(41, 128, 185)
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 16)
        pdf.set_y(8)
        pdf.cell(0, 10, "NeuroHR", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, "AI-Driven Cognitive Load & Workforce Performance Estimator", ln=True, align="C")
        pdf.set_font("Arial", "", 8)
        pdf.set_text_color(200, 200, 200)
        pdf.cell(0, 6, f"Report Generated: {r['date']}  |  Report ID: NHR-{datetime.now().strftime('%Y%m%d%H%M')}", ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(43)

        # Section 1: Employee Information
        pdf.set_fill_color(52, 73, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "  EMPLOYEE INFORMATION", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Employee info table
        pdf.set_font("Arial", "", 10)
        pdf.set_fill_color(240, 240, 240)
        col_w = pw / 2
        pdf.cell(col_w, 7, f"  Name: {r['name']}", border=1, fill=True)
        pdf.cell(col_w, 7, f"  Age: {r['age']} years", border=1, fill=True, ln=True)
        pdf.cell(col_w, 7, f"  Role Level: {r['role']}", border=1)
        pdf.cell(col_w, 7, f"  Department: {r['department']}", border=1, ln=True)
        pdf.cell(col_w, 6, f"  Project Phase: {r['project_phase']}", border=1, fill=True)
        pdf.cell(col_w, 6, f"  Employee Persona: {r['persona']}", border=1, fill=True, ln=True)
        pdf.ln(2)

        # Section 2: Input Parameters
        pdf.set_fill_color(52, 73, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "  INPUT PARAMETERS", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Parameters table
        pdf.set_font("Arial", "B", 9)
        pdf.set_fill_color(220, 220, 220)
        col_w3 = pw / 3
        pdf.cell(col_w3, 7, "  Parameter", border=1, fill=True, align="C")
        pdf.cell(col_w3, 7, "  Value", border=1, fill=True, align="C")
        pdf.cell(col_w3, 7, "  Range", border=1, fill=True, align="C", ln=True)

        pdf.set_font("Arial", "", 9)
        params = [
            ("Weekly Work Hours", f"{r['weekly_hours']} hrs", "20-100 hrs"),
            ("Overtime (30 days)", f"{r['overtime']} hrs", "0-120 hrs"),
            ("Active Tasks", str(r['active_tasks']), "1-50"),
            ("Stress Level", f"{r['stress']}/5", "1-5"),
            ("Performance Rating", f"{r['performance']}/5", "1-5"),
            ("Wellbeing Score", f"{r['wellbeing']}/100", "0-100"),
        ]
        for i, (param, val, rng) in enumerate(params):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(248, 248, 248)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_w3, 5, f"  {param}", border=1, fill=fill)
            pdf.cell(col_w3, 5, f"  {val}", border=1, fill=fill, align="C")
            pdf.cell(col_w3, 5, f"  {rng}", border=1, fill=fill, align="C", ln=True)
        pdf.ln(2)

        # Section 3: Analysis Results
        pdf.set_fill_color(52, 73, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "  ANALYSIS RESULTS", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Results in boxes
        col_w4 = pw / 4
        pdf.set_font("Arial", "B", 9)
        pdf.set_fill_color(41, 128, 185)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(col_w4, 6, "CLI Score", border=1, fill=True, align="C")
        pdf.cell(col_w4, 6, "Risk Level", border=1, fill=True, align="C")
        pdf.cell(col_w4, 6, "Burnout Prob.", border=1, fill=True, align="C")
        pdf.cell(col_w4, 6, "Perf. Impact", border=1, fill=True, align="C", ln=True)

        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 0, 0)

        # Color code risk
        if r['risk'] == "Critical":
            risk_color = (192, 57, 43)
        elif r['risk'] == "High":
            risk_color = (230, 126, 34)
        elif r['risk'] == "Medium":
            risk_color = (241, 196, 15)
        else:
            risk_color = (39, 174, 96)

        pdf.cell(col_w4, 10, f"{r['cli']:.3f}", border=1, align="C")
        pdf.set_fill_color(*risk_color)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(col_w4, 10, r['risk'], border=1, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_w4, 10, f"{r['burnout_prob']*100:.1f}%", border=1, align="C")
        pdf.cell(col_w4, 10, f"{r['delta']:.3f}", border=1, align="C", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", "", 9)
        pdf.set_fill_color(236, 240, 241)
        pdf.cell(0, 6, f"  Employee Profile Classification: {r['cluster_name']}", border=1, fill=True, ln=True)
        pdf.ln(2)

        # Section 4: HR Recommendation
        if r['risk'] == "Critical":
            rec_color = (192, 57, 43)
        elif r['risk'] == "High":
            rec_color = (230, 126, 34)
        else:
            rec_color = (39, 174, 96)

        pdf.set_fill_color(*rec_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "  HR RECOMMENDATION", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        if r['risk'] in ["Critical", "High"]:
            pdf.set_fill_color(255, 253, 231)
        else:
            pdf.set_fill_color(232, 245, 233)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, f"  Action: {r['recommendation']}", border=1, fill=True, ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5, f"  Details: {r['rec_desc']}", border=1, fill=True)
        pdf.ln(2)

        # Section 5: Risk Scale Reference
        pdf.set_fill_color(52, 73, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, "  CLI RISK SCALE REFERENCE", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Arial", "", 8)
        scales = [
            ("Low (0.00-0.30)", "Balanced, can handle more responsibility", (39, 174, 96)),
            ("Medium (0.30-0.55)", "Moderate stress, monitor closely", (241, 196, 15)),
            ("High (0.55-0.75)", "Fatigue signs, support needed", (230, 126, 34)),
            ("Critical (0.75-1.00)", "Burnout risk, immediate action", (192, 57, 43)),
        ]
        for level, desc, color in scales:
            pdf.set_fill_color(*color)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(38, 5, f" {level}", border=1, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_fill_color(255, 255, 255)
            pdf.cell(pw - 38, 5, f"  {desc}", border=1, ln=True)

        # CLI Explanation
        pdf.ln(3)
        pdf.set_font("Arial", "I", 7)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, "CLI (Cognitive Load Index) is a measure of mental workload stress (0-1). It is calculated using workload hours, stress levels, task frequency, sentiment, overtime, and other behavioral indicators. Burnout Probability estimates the likelihood of burnout based on CLI threshold (sigmoid function centered at 0.55).", align="L")

        # Footer
        pdf.ln(4)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, "This report is generated by NeuroHR - AI-Driven Cognitive Load Estimator", ln=True, align="C")
        pdf.cell(0, 5, "For internal HR use only. Confidential.", ln=True, align="C")

        return pdf.output(dest='S').encode('latin-1')

    pdf_bytes = generate_pdf()

    st.download_button(
        label="📄 Download Report (PDF)",
        data=pdf_bytes,
        file_name=f"NeuroHR_{r['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
