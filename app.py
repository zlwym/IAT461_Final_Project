#Imports

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Title

st.title("Boston Airbnb Neighbourhood Explorer")

#Load data

df = pd.read_csv("listings.csv")

#Clean price column

df["price"] = (
    df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

#Drop listings with no neighbourhood

df = df.dropna(
    subset=["neighbourhood_cleansed"]
).copy()

#Map 1: Most expensive neighbourhoods

st.header("1. Most Expensive Boston Neighbourhoods")

#Calculate average price for each neighbourhood

price_df = (
    df.groupby("neighbourhood_cleansed")
      .agg(
          average_price=("price", "mean"),
          listing_count=("price", "size"),
          latitude=("latitude", "mean"),
          longitude=("longitude", "mean")
      )
      .reset_index()
)

#Rename neighbourhood column

price_df = price_df.rename(
    columns={
        "neighbourhood_cleansed": "neighbourhood"
    }
)

#Map

price_map = px.scatter_map(
    price_df,
    lat="latitude",
    lon="longitude",
    color="average_price",
    size="listing_count",
    hover_name="neighbourhood",
    hover_data={
        "average_price": ":.2f",
        "listing_count": True,
        "latitude": False,
        "longitude": False
    },
    color_continuous_scale="Viridis",
    zoom=11,
    height=600,
    map_style="carto-darkmatter",
    title="Average Airbnb Price by Neighbourhood"
)

st.plotly_chart(
    price_map,
    width="stretch"
)


#Display most expensive neighbourhoods

st.subheader("Most Expensive Neighbourhoods")

expensive_df = (
    price_df
    .sort_values(
        "average_price",
        ascending=False
    )
    [["neighbourhood", "average_price", "listing_count"]]
)

expensive_df["average_price"] = (
    expensive_df["average_price"].round(2)
)

st.dataframe(
    expensive_df,
    width="stretch"
)

#Map 2: KMeans clustering

st.header("2. Airbnb Neighbourhood Clustering")

#Numeric features

features = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "number_of_reviews_ly",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

#Data on neighbourhoods

#Calculate the average value of every feature for each neighbourhood

neighbourhood_features = (
    df.groupby("neighbourhood_cleansed")[features]
      .mean()
)

#Remove neighbourhoods with missing values

neighbourhood_features = (
    neighbourhood_features
    .dropna()
)

#Scaling features

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    neighbourhood_features
)

#Slider for number of clusters

k = st.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=4
)


#Running KMeans

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    scaled_features
)

#PCA

pca = PCA(
    n_components=2
)

points = pca.fit_transform(
    scaled_features
)


pca_df = pd.DataFrame(
    points,
    columns=["PC1", "PC2"]
)

pca_df["cluster"] = clusters.astype(str)

pca_df["neighbourhood"] = (
    neighbourhood_features.index
)

#PCA vis

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="cluster",
    hover_name="neighbourhood",
    title="Boston Neighbourhood Similarity",
    labels={
        "cluster": "Cluster"
    }
)

st.plotly_chart(
    fig,
    width="stretch"
)

#Cluster map

#Get average location of each neighbourhood

centroid_df = (
    df.groupby("neighbourhood_cleansed")
      .agg(
          latitude=("latitude", "mean"),
          longitude=("longitude", "mean"),
          listing_count=("id", "size")
      )
      .reset_index()
)

#Keep only neighbourhoods used in clustering

centroid_df = centroid_df[
    centroid_df["neighbourhood_cleansed"]
    .isin(neighbourhood_features.index)
].copy()

#Match cluster labels to neighbourhoods

cluster_lookup = pd.DataFrame({
    "neighbourhood_cleansed":
        neighbourhood_features.index,
    "cluster":
        clusters.astype(str)
})

centroid_df = centroid_df.merge(
    cluster_lookup,
    on="neighbourhood_cleansed"
)

#Map

cluster_map = px.scatter_map(
    centroid_df,
    lat="latitude",
    lon="longitude",
    color="cluster",
    size="listing_count",
    hover_name="neighbourhood_cleansed",
    hover_data={
        "listing_count": True,
        "latitude": False,
        "longitude": False
    },
    zoom=11,
    height=600,
    map_style="carto-darkmatter",
    title="Boston Neighbourhoods by Airbnb Cluster"
)

st.plotly_chart(
    cluster_map,
    width="stretch"
)

#Clusters

st.subheader("Neighbourhoods in Each Cluster")

cluster_df = pd.DataFrame({
    "Neighbourhood":
        neighbourhood_features.index,
    "Cluster":
        clusters
})


for c in sorted(
    cluster_df["Cluster"].unique()
):

    st.write(
        f"### Cluster {c}"
    )

    members = cluster_df[
        cluster_df["Cluster"] == c
    ]

    st.write(
        list(members["Neighbourhood"])
    )