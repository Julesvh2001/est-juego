import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Comparación de Estilo de Juego")
st.title("Comparación de Estilo de Juego")

# Load dataset
@st.cache_data
def load_data():
    file_path = "Consolidado.xlsx"  # Ensure the file is in your working directory
    return pd.read_excel(file_path, sheet_name="Consolidado")

df = load_data()

# Sidebar filters
st.sidebar.title("Filter Options")

# **Multi-select for Leagues with "Select All" Option**
all_leagues = sorted(df["Liga"].dropna().astype(str).unique(), key=str)
selected_leagues = st.sidebar.multiselect("Select Leagues", ["Select All"] + all_leagues, default=["Select All"])

# If "Select All" is chosen, select all leagues automatically
if "Select All" in selected_leagues:
    selected_leagues = all_leagues

# **Dynamically filter available Seasons based on selected Leagues**
filtered_seasons = df[df["Liga"].isin(selected_leagues)]["Temporada"].dropna().astype(str).unique()
all_seasons = sorted(filtered_seasons, key=str)
selected_seasons = st.sidebar.multiselect("Select Seasons", ["Select All"] + all_seasons, default=["Select All"])

# If "Select All" is chosen, select all seasons automatically
if "Select All" in selected_seasons:
    selected_seasons = all_seasons

# **Dynamically filter available Técnicos based on selected Seasons**
filtered_tecnicos = df[df["Temporada"].astype(str).isin(selected_seasons)]["Técnico"].dropna().astype(str).unique()
all_tecnicos = sorted(filtered_tecnicos, key=str)
selected_tecnicos = st.sidebar.multiselect("Select Técnicos", ["Select All"] + all_tecnicos, default=["Select All"])

# If "Select All" is chosen, select all técnicos automatically
if "Select All" in selected_tecnicos:
    selected_tecnicos = all_tecnicos

# **Apply Filters for PCA & Clustering**
df_filtered_for_pca = df[df["Temporada"].astype(str).isin(selected_seasons)]  # Keep all técnicos for PCA
df_display = df_filtered_for_pca[df_filtered_for_pca["Técnico"].isin(selected_tecnicos)]  # Show only selected técnico(s)

# Check if there's data to display
if df_display.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# Select key metrics for clustering
features = [
    "Possession%", "Directness", "PPDA", "Deep Progressions", 
    "Deep Progressions Conceded", "OBV", "Shots", "Shots Conceded"
]

# Standardize and Apply PCA on all data (not just selected técnicos)
X = df_filtered_for_pca[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
df_filtered_for_pca["PCA1"], df_filtered_for_pca["PCA2"] = pca.fit_transform(X_scaled).T  

# Apply K-Means clustering to the entire dataset (not just the selected técnicos)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_filtered_for_pca["Cluster"] = kmeans.fit_predict(X_scaled)

# **Ensure clusters always have the same meaning**
# Get the mean values for each cluster to map them correctly
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
cluster_order = cluster_centers.mean(axis=1).sort_values().index  # Sort clusters consistently
cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
df_filtered_for_pca["Cluster"] = df_filtered_for_pca["Cluster"].map(cluster_mapping)

# **Define cluster descriptions based on fixed mapping**
cluster_descriptions = {
    "0": "Estilo de Juego Balanceado",
    "1": "Poca Posesión y Endeblidad Defensiva",
    "2": "Juego de Posesión, Alta Intensidad",
    "3": "Solidez Defensiva y Juego Directo",
}

# **Apply the same fixed colors regardless of técnico selection**
color_mapping = {
    "0": "#1f77b4",  # Blue
    "1": "#d62728",  # Red
    "2": "#ff9896",  # Light Red
    "3": "#aec7e8",  # Light Blue
}

# Update the graph to use consistent cluster names
df_display = df_display.merge(df_filtered_for_pca[["Técnico", "Equipo", "PCA1", "PCA2", "Cluster"]], 
                              on=["Técnico", "Equipo"], how="left")

fig = px.scatter(
    df_display,
    x="PCA1", 
    y="PCA2",
    color=df_display["Cluster"].astype(str),  
    category_orders={"Cluster": list(cluster_descriptions.keys())},
    color_discrete_map=color_mapping,
    hover_data={
        "Técnico": True, 
        "Equipo": True, 
        "Temporada": True, 
        "PCA1": False,  
        "PCA2": False,  
          
    },
    title=f"PCA Clustering: {', '.join(selected_leagues)} - {', '.join(selected_seasons)}",
    labels={"PCA1": "Control vs. Reactive Play", "PCA2": "Directness & Pressing Intensity"},
)

# Ensure the legend always displays descriptions instead of numbers
fig.for_each_trace(lambda t: t.update(name=f"{t.name}: {cluster_descriptions[t.name]}"))

# **Add black dashed reference lines to divide the graph into four quadrants**
fig.add_shape(
    type="line", x0=min(df_filtered_for_pca["PCA1"]), x1=max(df_filtered_for_pca["PCA1"]),
    y0=0, y1=0, line=dict(color="white", width=2, dash="dash")
)
fig.add_shape(
    type="line", x0=0, x1=0, 
    y0=min(df_filtered_for_pca["PCA2"]), y1=max(df_filtered_for_pca["PCA2"]),
    line=dict(color="white", width=2, dash="dash")
)

fig.update_layout(width=800, height=800, xaxis=dict(scaleanchor="y"), yaxis=dict(scaleanchor="x"))

# Show plot in Streamlit
st.plotly_chart(fig)
