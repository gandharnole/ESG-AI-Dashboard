import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# Load the model
try:
    model = tf.keras.models.load_model("esg_model.h5")
except FileNotFoundError as e:
    st.error(f"Error: Model file not found. Please ensure 'esg_model.h5' is in the correct location. {e}")
    st.stop()

def predict_risk_level(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

def get_esg_suggestions(input_data, prediction):
    suggestions = []
    environmental_threshold = 0.6
    social_threshold = 0.5
    governance_threshold = 0.4

    if prediction[0][0] > environmental_threshold or input_data[0] > 70:
        suggestions.append("🌱 Reduce carbon footprint by adopting renewable energy sources and optimizing logistics.")
    if prediction[0][1] < environmental_threshold or input_data[1] < 50:
        suggestions.append("💡 Improve energy efficiency with smart automation and LED lighting solutions.")
    if prediction[0][2] < environmental_threshold or input_data[2] < 50:
        suggestions.append("♻️ Implement advanced waste management and recycling programs.")
    if prediction[0][3] < environmental_threshold or input_data[3] < 50:
        suggestions.append("🧪 Use more Eco-Friendly materials in production.")

    if prediction[0][4] < social_threshold or input_data[4] < 50:
        suggestions.append("🦺 Enhance worker safety training and provide high-quality PPE equipment.")
    if prediction[0][5] < social_threshold or input_data[5] < 50:
        suggestions.append("👩‍💼 Promote diversity and inclusion through equitable hiring and promotion practices.")
    if prediction[0][6] < social_threshold or input_data[6] < 50:
        suggestions.append("🤝 Strengthen CSR initiatives to engage with local communities and stakeholders.")

    if prediction[0][7] < governance_threshold or input_data[7] < 50:
        suggestions.append("📜 Strengthen policy compliance with regular audits and risk assessments.")
    if prediction[0][8] < governance_threshold or input_data[8] < 50:
        suggestions.append("🔍 Improve transparency by publishing detailed sustainability reports and disclosures.")
    if prediction[0][9] < governance_threshold or input_data[9] < 50:
        suggestions.append("📈 Improve Risk Management strategies.")

    return suggestions

def create_radar_chart(input_data):
    categories = ['Carbon Emissions', 'Energy Efficiency', 'Waste Management', 'Eco-Friendly Materials',
                  'Worker Safety', 'Diversity & Inclusion', 'CSR Activities', 'Policy Compliance', 'Transparency', 'Risk Management']

    fig = go.Figure(go.Scatterpolar(
        r=input_data + [input_data[0]],
        theta=categories + [categories[0]],
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    return fig

def create_bar_chart(prediction):
    input_names = ['Carbon Emissions', 'Energy Efficiency', 'Waste Management', 'Eco-Friendly Materials', 'Worker Safety', 'Diversity & Inclusion', 'CSR Activities', 'Policy Compliance', 'Transparency', 'Risk Management']
    fig = go.Figure(data=[go.Bar(x=input_names, y=prediction[0])])
    fig.update_layout(title='Input Probability')
    return fig

# Streamlit UI
st.set_page_config(page_title="ESG Risk Dashboard", layout="wide")
st.title("🌍 ESG Risk Assessment Dashboard")
st.markdown("Analyze ESG parameters and get recommendations to improve your **sustainability score**.")

# User Inputs (Organized in columns)
st.sidebar.header("Enter ESG Parameters:")
col1, col2 = st.sidebar.columns(2)
carbon_emissions = col1.slider("Carbon Emissions (Metric Tons)", 0, 100, 50)
energy_efficiency = col2.slider("Energy Efficiency (%)", 0, 100, 50)
waste_management = col1.slider("Waste Management (%)", 0, 100, 50)
eco_friendly_materials = col2.slider("Eco-Friendly Materials (%)", 0, 100, 50)
worker_safety = col1.slider("Worker Safety (%)", 0, 100, 50)
diversity_inclusion = col2.slider("Diversity & Inclusion (%)", 0, 100, 50)
csr_activities = col1.slider("CSR Activities (%)", 0, 100, 50)
policy_compliance = col2.slider("Policy Compliance (%)", 0, 100, 50)
transparency_score = col1.slider("Transparency Score (%)", 0, 100, 50)
risk_management = col2.slider("Risk Management (%)", 0, 100, 50)

input_data = [carbon_emissions, energy_efficiency, waste_management, eco_friendly_materials,
              worker_safety, diversity_inclusion, csr_activities, policy_compliance,
              transparency_score, risk_management]

if st.sidebar.button("🔍 Predict Risk Level"):
    prediction = predict_risk_level(input_data)

    col3, col4 = st.columns(2)
    col3.plotly_chart(create_radar_chart(input_data), use_container_width=True)
    col4.plotly_chart(create_bar_chart(prediction), use_container_width=True)

    st.write("Prediction Probabilities of each Input:")
    input_names = ['Carbon Emissions', 'Energy Efficiency', 'Waste Management', 'Eco-Friendly Materials', 'Worker Safety', 'Diversity & Inclusion', 'CSR Activities', 'Policy Compliance', 'Transparency', 'Risk Management']
    for i, prob in enumerate(prediction[0]):
        st.write(f"- {input_names[i]}: {prob:.4f}")

    st.subheader("💡 Suggested Improvements")
    suggestions = get_esg_suggestions(input_data, prediction)
    if suggestions:
        for suggestion in suggestions:
            st.write(f"✅ {suggestion}")
    else:
        st.write("✔️ Your ESG score is excellent! Keep up the great work.")