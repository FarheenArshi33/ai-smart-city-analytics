import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page Configuration
st.set_page_config(
    page_title="Smart City Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Professional UI Styling
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4FFF8;
}

/* Titles */
h1 {
    color: #00695C;
    font-size: 42px;
    font-weight: bold;
}

h2, h3 {
    color: #00796B;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #DFF5E1;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 2px solid #B2DFDB;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
}

/* Text visibility */
html, body, [class*="css"] {
    color: #222222;
    font-size: 16px;
}

/* Buttons */
.stButton>button {
    background-color: #009688;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #00796B;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")

# Load ML model
model = joblib.load("models/aqi_model.pkl")

# Dashboard Title
st.title("🌍 Smart City Air Pollution Analytics Dashboard")

st.markdown("""
### Real-time Pollution Insights, Geo Analytics & AQI Prediction
""")

# Sidebar
st.sidebar.header("📌 Dashboard Filters")

# Numeric columns
numeric_columns = df.select_dtypes(include='number').columns

# Metric selection
selected_column = st.sidebar.selectbox(
    "Select Pollution Metric",
    numeric_columns
)

# Country selection
countries = sorted(df["Country"].unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries
)

# Filter dataset
filtered_df = df[df["Country"] == selected_country]

# KPI Metrics
st.subheader("📊 Key Pollution Insights")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Value",
    round(filtered_df[selected_column].mean(), 2)
)

col2.metric(
    "Maximum Value",
    round(filtered_df[selected_column].max(), 2)
)

col3.metric(
    "Minimum Value",
    round(filtered_df[selected_column].min(), 2)
)

# Dataset Preview
st.subheader("📄 Dataset Preview")

st.dataframe(filtered_df.head())

# Line Chart
st.subheader(f"📈 {selected_column} Trend Analysis")

fig1 = px.line(
    filtered_df,
    y=selected_column,
    template="plotly_white"
)

st.plotly_chart(fig1, use_container_width=True)

# Histogram
st.subheader(f"📊 Distribution of {selected_column}")

fig2 = px.histogram(
    filtered_df,
    x=selected_column,
    nbins=30,
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True)

# Top 10 Cities
st.subheader(f"🏙️ Top 10 Most Polluted Cities")

city_pollution = filtered_df.groupby("City")[selected_column].mean().reset_index()

top_cities = city_pollution.nlargest(10, selected_column)

fig3 = px.bar(
    top_cities,
    x="City",
    y=selected_column,
    color=selected_column,
    template="plotly_white"
)

st.plotly_chart(fig3, use_container_width=True)

# Country Comparison Map
st.subheader("🌍 Global Pollution Map")

country_avg = df.groupby("Country")[selected_column].mean().reset_index()

fig_map = px.choropleth(
    country_avg,
    locations="Country",
    locationmode="country names",
    color=selected_column,
    hover_name="Country",
    color_continuous_scale="YlGnBu",
    template="plotly_white"
)

st.plotly_chart(fig_map, use_container_width=True)

# Heatmap
st.subheader("🔥 Correlation Heatmap")

corr = df[numeric_columns].corr()

fig4 = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    template="plotly_white"
)

st.plotly_chart(fig4, use_container_width=True)

# AQI Prediction
st.subheader("🤖 AQI Prediction Using Machine Learning")

co = st.number_input("CO AQI Value", min_value=0.0)
ozone = st.number_input("Ozone AQI Value", min_value=0.0)
no2 = st.number_input("NO2 AQI Value", min_value=0.0)
pm25 = st.number_input("PM2.5 AQI Value", min_value=0.0)

if st.button("Predict AQI"):

    prediction = model.predict([[
        co,
        ozone,
        no2,
        pm25
    ]])

    st.success(f"✅ Predicted AQI Value: {round(prediction[0], 2)}")

# Country Statistics
st.subheader("🌎 Country Statistics")

st.write(filtered_df.describe())

# Footer
st.markdown("""
---
### 🚀 Developed using Python, Streamlit, Plotly & Machine Learning
""")