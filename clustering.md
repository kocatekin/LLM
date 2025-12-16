# Clustering

We are working with mall customers data on kaggle: https://www.kaggle.com/datasets/shwetabh123/mall-customers


Google Colab runs our code on a remote machine, not on our personal computer.
Because of this, files that exist on our local computer are not automatically visible to Colab.

To work with our dataset, we must explicitly upload it into the Colab runtime.
The uploaded file will be stored temporarily:

* It exists only while the notebook session is active
* If the runtime resets, the file must be uploaded again


```python
from google.colab import files

uploaded = files.upload()
```

---

Before we begin, we import the libraries needed for:
* data handling
* visualization
* machine learning algorithms

```python
# -----------------------------
# 1. Imports
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
```

---

## Loading and Understanding the Data

Now that the CSV file is inside the Colab runtime, we need a tool to work with tabular data.
In Python, the standard library for this is **pandas**.

We load the CSV file into a DataFrame, which is a 2-dimensional data structure similar to a table:

* rows = observations (customers)
* columns = features (age, income, spending score, etc)

At this stage, our goal is **not modeling**. Our goal is to:

* confirm the file loaded correctly
* understand what data we have


For clustering, we focus on **numerical variables** that describe behavior rather than identity.

In this example, we use:

* Annual Income
* Spending Score

These features allow us to compare customers in terms of purchasing power and shopping behavior.

```python
# -----------------------------
# 2. Load Data
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
```

When this is run, we will see the first few columns and we can verify the column names and basic structure.

## Inspecting the dataset structure

Before applying any machine learning algorithm, we must understand the structure of the dataset.

Clustering algorithms operate on numerical feature spaces, so we need to identify:

* how many observations we have
* which columns are numeric
* which columns are identifiers or non-numerics

We will use:
* `df.shape`
* `df.columns`
* `df.info()`

Here;
* each row is a customer
* `CustomerID` is an *identifier*, not a feature.  
* `Gender` is categorical.
* Remaining columns are numeric and suitable for clustering.


---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis helps us build intuition about the dataset before applying algorithms.

For clustering in particular, EDA helps us:

* understand feature distributions
* identify natural groupings
* decide which variables might define similarity.

At this EDA stage we:
* **do not** transform data
* **do not** scale
* only visualize and summarize


### Basic Statistical Summary

This gives a high-level view of ranges and variability.

```python
df.describe()
```

* Different columns have different scales
* Spending score is bounded (1-100)
* Income has a wider numeric range

### Distribution of Annual Income

```python
import matplotlib.pyplot as plt

plt.hist(df["Annual Income (k$)"], bins=20)
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.show()
```

This shows:
* spread of income levels
* presence or absence of extreme values

### Distribution of Spending Score

```python
plt.hist(df["Spending Score (1-100)"], bins=20)
plt.title("Distribution of Spending Score")
plt.xlabel("Spending Score (1–100)")
plt.ylabel("Number of Customers")
plt.show()
```

Shows:
* bounded nature of the variable
* behavioral interpretation (low vs. high spenders)

### Income vs. Spending Relationship

```python
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)
plt.title("Annual Income vs. Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.show()
```

Shows:
* visible dense regions
* intuition for customer segmentation
* why clustering is appropriate

Based on what we see, income and spending score are good candidates for defining similarity between customers.

## Selecting features for clustering


Based on our exploratory analysis, we now make an explicit modeling decision.

Clustering requires us to define what similarity means.
Here, we define similarity between customers using:

* purchasing power (annual income)
* purchasing behavior (spending score)

We exclude:
* `CustomerID`
* `Gender` -> so we don't encode now.

This step converts our full dataset into a feature matrix used by the clustering algorithm.

```python
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X.head()
```

At this point:
* `X` is a 2D numerical space
* each row is a customer
* distance between rows represent similarity

---


## Why Feature Scaling Is Necessary

Most clustering algorithms rely on distance calculations.

If features are on different scales:

* larger-scale features dominate the distance computation
* clustering results become misleading


If one feature has larger numeric values than another, it will dominate the distance computation, even if it is not more important.

To prevent this, we standardize the features so they contribute equally.

```python
# -----------------------------
# 3. Scale Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## K-Means Clustering: Trying Different Numbers of Clusters

K-Means requires us to choose the number of clusters (**k**) in advance.

There is no single “correct” value, so we explore multiple options.

Here, we compare:

* a coarse segmentation (k = 3)
* a more detailed segmentation (k = 5)




```python
# -----------------------------
# 4. K-Means (k = 3, 5)
# -----------------------------
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_5 = KMeans(n_clusters=5, random_state=42)

df["KMeans_3"] = kmeans_3.fit_predict(X_scaled)
df["KMeans_5"] = kmeans_5.fit_predict(X_scaled)
```

---

## Visualizing K-Means Results

Visualization helps us understand how the algorithm grouped customers.

* Each point is a customer.
* Colors indicate cluster membership.


```python
# -----------------------------
# 5. Visualize K-Means Results
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_3"]
)
axes[0].set_title("K-Means (k=3)")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score")

axes[1].scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_5"]
)
axes[1].set_title("K-Means (k=5)")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score")

plt.show()
```
---


## Agglomerative

Agglomerative clustering takes a different approach:

* it starts with each customer as its own cluster
* clusters are merged step by step based on similarity

Unlike K-Means, there are:

* no centroids
* no iterative reassignment

We still specify the number of final clusters.

```python
# -----------------------------
# 6. Agglomerative Clustering
# -----------------------------
agg = AgglomerativeClustering(n_clusters=5)
df["Agglomerative"] = agg.fit_predict(X_scaled)
```

---

## Visualizing Agglomerative Clustering

Even with the same number of clusters, different algorithms can produce different segment boundaries.

```python
# -----------------------------
# 7. Visualize Agglomerative Clustering
# -----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Agglomerative"]
)
plt.title("Agglomerative Clustering (k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()
```

---

## DBSCAN: Density-Based Clustering

DBSCAN groups customers based on dense regions rather than fixed cluster counts.

Key differences:

* no need to specify k
* can identify outliers (noise)
* sensitive to parameter choices

```python
# -----------------------------
# 8. DBSCAN
# -----------------------------
dbscan = DBSCAN(eps=0.6, min_samples=5)
df["DBSCAN"] = dbscan.fit_predict(X_scaled)
```

---

## Visualizing DBSCAN Results

Points labeled **-1** are considered noise by the algorithm.

```python
# -----------------------------
# 9. Visualize DBSCAN
# -----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["DBSCAN"]
)
plt.title("DBSCAN Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()
```

---

## Interpreting the Clusters

Clustering algorithms do not assign meaning.

We create meaning by analyzing cluster characteristics.

One simple approach is to compute average feature values per cluster.

```python
# -----------------------------
# 10. Compare Cluster Means (K-Means k=5)
# -----------------------------
df.groupby("KMeans_5")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
```

This allows us to describe clusters qualitatively, such as:

* high income / high spending
* high income / low spending
* low income / high spending


“Clustering gives us **structure**.
Humans give it **meaning**.”





