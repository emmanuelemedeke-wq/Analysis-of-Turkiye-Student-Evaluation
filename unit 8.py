from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset from the same folder as this script
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "turkiye-student.csv"
df = pd.read_csv(csv_path)

# -----------------------------
# Basic dataset check
# -----------------------------
print("DATASET SHAPE:", df.shape)
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nDATA TYPES AND NON-NULL COUNTS:")
df.info()
print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())
print("\nMISSING VALUES:")
print(df.isnull().sum())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
# Use evaluation question columns only.
# In this dataset, columns from index 3 onward are rating variables.
X = df.iloc[:, 3:].copy()

# Figure 1: Histogram
plt.figure(figsize=(8, 5))
plt.hist(df.iloc[:, 3], bins=5, edgecolor='black')
plt.title("Histogram of Course Evaluation Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()


# Figure 2: Box Plot
plt.figure(figsize=(6, 5))
plt.boxplot(df.iloc[:, 3])
plt.title("Box Plot of Evaluation Scores")
plt.ylabel("Score")
plt.tight_layout()


# Figure 3: Bar Chart
plt.figure(figsize=(7, 5))
df.iloc[:, 3].value_counts().sort_index().plot(kind="bar")
plt.title("Bar Chart of Response Distribution")
plt.xlabel("Rating (1-5)")
plt.ylabel("Count")
plt.tight_layout()


# Figure 4: Scatter Plot
plt.figure(figsize=(7, 5))
plt.scatter(df.iloc[:, 3], df.iloc[:, 4], alpha=0.5)
plt.title("Scatter Plot Between Two Evaluation Questions")
plt.xlabel("Question 1")
plt.ylabel("Question 2")
plt.tight_layout()


# -----------------------------
# Standardize data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaling complete.")

# -----------------------------
# Elbow Method
# -----------------------------
inertia = []
for k in range(1, 10):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()


# -----------------------------
# Final K-Means model
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Add cluster labels to original data
df["Cluster"] = kmeans.labels_

# Cluster counts
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# Figure 5 / output table
print("\nDATASET WITH CLUSTER COLUMN:")
print(df.head())


#Show all figures at the same time
plt.show()

# Cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
print("\nCLUSTER CENTERS (STANDARDIZED SCALE):")
print(cluster_centers)


# -----------------------------
# Silhouette Score
# -----------------------------
score = silhouette_score(X_scaled, kmeans.labels_)
print("\nSILHOUETTE SCORE:", score)

# -----------------------------
# Cluster mean summary table
# -----------------------------
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print("\nCLUSTER MEAN SUMMARY TABLE:")
print(cluster_summary)
