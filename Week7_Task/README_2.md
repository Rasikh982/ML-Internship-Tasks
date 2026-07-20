# Customer Segmentation App

This is a simple Python project that groups mall customers into segments
(clusters) based on their age, income, and spending behavior — using K-Means
clustering.

## Files

- `app.py` - main code file
- `Mall_Customers.csv` - dataset used for clustering
- `README.md` - this file

When you run the code, it will also create these output files:
- `elbow_plot.png` - graph to find the best number of clusters
- `clusters_pca.png` - 2D graph showing the customer clusters
- `customers_with_clusters.csv` - original data with a new "Cluster" column

## Dataset

The dataset has 200 rows and these columns:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## How the code works

1. Load the dataset using pandas
2. Select useful columns: Age, Annual Income, Spending Score
3. Scale the data (make all numbers on the same range)
4. Use the **Elbow Method** to find the best number of clusters
   - This is saved as `elbow_plot.png`
   - Look for the "elbow" point (where the line bends) in the graph
5. Train a **K-Means** model (default is 5 clusters, common for this dataset)
6. Check cluster quality using **Silhouette Score**
   - Closer to 1 = better separated clusters
7. Use **PCA** to reduce data to 2D and visualize the clusters
   - This is saved as `clusters_pca.png`
8. Create a **cluster profile** (average Age, Income, Spending Score per group)
   so you can understand what each customer group looks like
   (for example: "high income, low spending" or "young, high spending")

## How to run

1. Install the required libraries:

```
pip install pandas scikit-learn matplotlib
```

2. Make sure `app.py` and `Mall_Customers.csv` are in the same folder.

3. Run the code:

```
python app.py
```

## Notes

- K-Means is used here because it is simple and widely used for customer
  segmentation.
- You can also try **DBSCAN** or **Hierarchical Clustering** by changing the
  model in `app.py` (import from `sklearn.cluster`).
- The number of clusters is set to 5 by default. Change the `best_k` variable
  in `app.py` after looking at the elbow plot if you want a different number.
