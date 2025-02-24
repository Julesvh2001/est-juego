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

# **Define key metrics before filtering**
features = [
    "Possession%", "Directness", "PPDA", "Deep Progressions", 
    "Deep Progressions Conceded", "OBV", "Shots", "Shots Conceded"
]

# Sidebar filters
st.sidebar.title("Filter Options")

# **Multi-select for Leagues with "Select All" Option**
all_leagues = sorted(df["Liga"].dropna().astype(str).unique(), key=str)
selected_leagues = st.sidebar.multiselect("Select Leagues", ["Select All"] + all_leagues, default=["Select All"])

# If "Select All" is chosen, select all leagues automatically
if "Select All" in selected_leagues:
    selected_leagues = all_leagues

# **Dynamically filter available Seasons based on selected Leagues**
df_league_filtered = df[df["Liga"].isin(selected_leagues)]  # <-- Apply League Filter
filtered_seasons = df_league_filtered["Temporada"].dropna().astype(str).unique()
all_seasons = sorted(filtered_seasons, key=str)
selected_seasons = st.sidebar.multiselect("Select Seasons", ["Select All"] + all_seasons, default=["Select All"])

# If "Select All" is chosen, select all seasons automatically
if "Select All" in selected_seasons:
    selected_seasons = all_seasons

# **Dynamically filter available Técnicos based on selected Seasons**
df_season_filtered = df_league_filtered[df_league_filtered["Temporada"].astype(str).isin(selected_seasons)]  # <-- Apply Season Filter
filtered_tecnicos = df_season_filtered["Técnico"].dropna().astype(str).unique()
all_tecnicos = sorted(filtered_tecnicos, key=str)
selected_tecnicos = st.sidebar.multiselect("Select Técnicos", ["Select All"] + all_tecnicos, default=["Select All"])

# If "Select All" is chosen, select all técnicos automatically
if "Select All" in selected_tecnicos:
    selected_tecnicos = all_tecnicos

# **Apply Final Filtering and Keep "Cluster" Column**
df_filtered = df_season_filtered[df_season_filtered["Técnico"].isin(selected_tecnicos)].copy()  # <-- Ensure copy()
df_filtered = df_filtered[["Liga", "Temporada", "Técnico", "Equipo"] + features]  # <-- Keep relevant columns

# **Check if filtered data is empty**
if df_filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# **Compute PCA and Clustering Once on the Full Dataset**
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
df["PCA1"], df["PCA2"] = pca.fit_transform(X_scaled).T  

# Apply K-Means clustering **once**
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# **Ensure clusters always have the same meaning**
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
cluster_order = cluster_centers.mean(axis=1).sort_values().index  # Sort clusters consistently
cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
df["Cluster"] = df["Cluster"].map(cluster_mapping)

# **Now filter the dataset for visualization (with Cluster)**
df_display = df_filtered.merge(df[["Liga", "Temporada", "Técnico", "Equipo", "PCA1", "PCA2", "Cluster"]], 
                               on=["Liga", "Temporada", "Técnico", "Equipo"], how="left")

# **Check if Cluster column is present**
if "Cluster" not in df_display.columns:
    st.error("Error: Cluster column is missing after filtering. Please check merge operation.")
    st.stop()

# **Define cluster descriptions**
cluster_descriptions = {
    0: "Estilo de Juego Balanceado",
    1: "Poca Posesión y Endeblidad Defensiva",
    2: "Juego de Posesión, Alta Intensidad",
    3: "Solidez Defensiva y Juego Directo",
}


# **Update the graph to use consistent cluster names**
df_display["Cluster Name"] = df_display["Cluster"].map(cluster_descriptions)

fig = px.scatter(
    df_display,
    x="PCA1", 
    y="PCA2",
    color="Cluster Name",  
    category_orders={"Cluster Name": list(cluster_descriptions.values())},
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

# **Add black dashed reference lines to divide the graph into four quadrants**
fig.add_shape(
    type="line", x0=min(df["PCA1"]), x1=max(df["PCA1"]),
    y0=0, y1=0, line=dict(color="white", width=2, dash="dash")
)
fig.add_shape(
    type="line", x0=0, x1=0, 
    y0=min(df["PCA2"]), y1=max(df["PCA2"]),
    line=dict(color="white", width=2, dash="dash")
)

fig.update_layout(width=800, height=800, xaxis=dict(scaleanchor="y"), yaxis=dict(scaleanchor="x"))

# Show plot in Streamlit
st.plotly_chart(fig)
