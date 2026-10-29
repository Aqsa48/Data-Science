"""
Machine Learning Basics for Data Science
==========================================
Covers: Linear Regression, Logistic Regression, Decision Tree, K-Means Clustering,
        train/test split, cross-validation, and model evaluation metrics.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

# reproducibility
SEED = 42
np.random.seed(SEED)

# =============================================================
# 1. LINEAR REGRESSION
# =============================================================
print("=" * 55)
print("1. Linear Regression")
print("=" * 55)

X_reg, y_reg = make_regression(n_samples=200, n_features=3,
                                noise=15, random_state=SEED)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=SEED
)

lr = LinearRegression()
lr.fit(X_train_r, y_train_r)
y_pred_r = lr.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

print(f"Intercept : {lr.intercept_:.4f}")
print(f"Coefficients: {lr.coef_.round(4)}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Cross-validation
cv_r2 = cross_val_score(LinearRegression(), X_reg, y_reg, cv=5, scoring="r2")
print(f"5-fold CV R²: {cv_r2.round(4)}  mean={cv_r2.mean():.4f}")

# =============================================================
# 2. LOGISTIC REGRESSION
# =============================================================
print("\n" + "=" * 55)
print("2. Logistic Regression")
print("=" * 55)

X_cls, y_cls = make_classification(n_samples=300, n_features=5,
                                    n_informative=3, random_state=SEED)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=SEED
)

log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(random_state=SEED, max_iter=200))
])

log_pipe.fit(X_train_c, y_train_c)
y_pred_c = log_pipe.predict(X_test_c)

print(f"Accuracy : {accuracy_score(y_test_c, y_pred_c):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_c))
print("Classification Report:\n", classification_report(y_test_c, y_pred_c))

cv_acc = cross_val_score(log_pipe, X_cls, y_cls, cv=5, scoring="accuracy")
print(f"5-fold CV Accuracy: {cv_acc.round(4)}  mean={cv_acc.mean():.4f}")

# =============================================================
# 3. DECISION TREE CLASSIFIER
# =============================================================
print("\n" + "=" * 55)
print("3. Decision Tree Classifier")
print("=" * 55)

dt = DecisionTreeClassifier(max_depth=4, random_state=SEED)
dt.fit(X_train_c, y_train_c)
y_pred_dt = dt.predict(X_test_c)

print(f"Accuracy : {accuracy_score(y_test_c, y_pred_dt):.4f}")
print("Feature importances:", dt.feature_importances_.round(4))
print("Tree depth:", dt.get_depth())
print("Leaf nodes:", dt.get_n_leaves())

cv_dt = cross_val_score(dt, X_cls, y_cls, cv=5, scoring="accuracy")
print(f"5-fold CV Accuracy: {cv_dt.round(4)}  mean={cv_dt.mean():.4f}")

# =============================================================
# 4. K-MEANS CLUSTERING
# =============================================================
print("\n" + "=" * 55)
print("4. K-Means Clustering")
print("=" * 55)

X_blob, y_true = make_blobs(n_samples=300, n_features=2,
                             centers=4, cluster_std=1.0, random_state=SEED)

# Choose k via inertia (elbow method values)
inertias = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_blob)
    inertias.append(km.inertia_)
print("Inertia for k=2..7:", [round(v, 1) for v in inertias])

# Fit with best k=4
km4 = KMeans(n_clusters=4, random_state=SEED, n_init=10)
labels4 = km4.fit_predict(X_blob)

sil = silhouette_score(X_blob, labels4)
print(f"\nK=4 Silhouette Score: {sil:.4f}")
print("Cluster centers:\n", km4.cluster_centers_.round(3))
print("Cluster sizes:", {k: int((labels4 == k).sum()) for k in range(4)})

# =============================================================
# 5. MODEL EVALUATION SUMMARY
# =============================================================
print("\n" + "=" * 55)
print("5. Model Comparison Summary")
print("=" * 55)

results = {
    "Logistic Regression": cv_acc.mean(),
    "Decision Tree":       cv_dt.mean(),
}
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<28}  CV Accuracy = {score:.4f}")

print("\nDone! Machine Learning basics covered successfully.")
