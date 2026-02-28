import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("Customer Segmentation using K-Means")
st.write("Identify Customer groups for targeted marketing.")

# load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

st.subheader("Dataset preview")
st.dataframe(df.head())

#Feature selection
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar for Cluster selection
st.sidebar.header("Model Configuration")
k = st.sidebar.slider("Select Number of Cluster (K)", 2, 10, 5)

# Train KMeans
kmeans = KMeans(n_clusters=k, random_state=42) # type: ignore
clusters = kmeans.fit_predict(X_scaled) # type: ignore

df['Cluster'] = clusters

# Silhouette Score
score = silhouette_score(X_scaled, clusters)

st.sidebar.write(f"Silhouette Score : {round(score, 3)}")

# Visualization
st.subheader("Cluster Visualization")

fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(x = "Annual Income (k$)", y = "Spending Score (1-100)",hue = 'Cluster', palette = 'Set1', data = df, ax = ax)
plt.title("Customer Segments")
st.pyplot(fig)

# Cluster Summary
st.subheader("Cluster Summary")

cluster_summary = df.groupby("Cluster")[features].mean()
st.subheader("Predict New Customer Segment")

income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending = st.number_input("Spending score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Segment"):
    new_customer = np.array([[income, spending]])
    new_scaled = scaler.transform(new_customer)
    prediction = kmeans.predict(new_scaled) # type: ignore

    st.success(f"Predicted Customer Segment: {prediction[0]}")

    if prediction[0] == 0:
        st.info("High Value Customer")
    elif prediction[0] == 1:
        st.info("Low Spending Group")
    else:
        st.info("Moderate Customer")

st.markdown("----")
st.write("Developed by Varsha Patil")