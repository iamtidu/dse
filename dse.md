## Clean a given dataset to handle missing values using methods like dropping rows, replacing with mean, median, or mode, and demonstrate your approach.

```python
import pandas as pd
import numpy as np

# Step 1: Create a Sample Dataset
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", np.nan],
    "Age": [25, np.nan, 30, 29, 35],
    "Salary": [50000, 60000, np.nan, 52000, 58000],
    "Department": ["HR", "IT", "Finance", "Marketing", np.nan]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Step 2: Drop Rows with Missing Values
df_dropped = df.dropna()
print("\nDataset After Dropping Rows with Missing Values:")
print(df_dropped)

# Step 3: Replace Missing Values with Mean (For Numerical Columns)
df_mean = df.copy()
df_mean["Age"] = df_mean["Age"].fillna(df_mean["Age"].mean())
df_mean["Salary"] = df_mean["Salary"].fillna(df_mean["Salary"].mean())
print("\nDataset After Replacing Missing Values with Mean:")
print(df_mean)

# Step 4: Replace Missing Values with Median (For Numerical Columns)
df_median = df.copy()
df_median["Age"] = df_median["Age"].fillna(df_median["Age"].median())
df_median["Salary"] = df_median["Salary"].fillna(df_median["Salary"].median())
print("\nDataset After Replacing Missing Values with Median:")
print(df_median)

# Step 5: Replace Missing Values with Mode (For Categorical Columns)
df_mode = df.copy()
for column in ["Name", "Department"]:
    df_mode[column] = df_mode[column].fillna(df_mode[column].mode()[0])
print("\nDataset After Replacing Missing Values with Mode:")
print(df_mode)
```

## Perform exploratory data analysis (EDA) on a student performance dataset. Identify key patterns, relationships, and summarize the dataset using descriptive statistics.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
# Replace 'student_performance.csv' with your actual dataset path
df = pd.read_csv("student_performance.csv")

# Step 2: Overview of the Dataset
print("Dataset Overview:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Handle Missing Values (if any)
# Replace numerical missing values with the column mean
df.fillna(df.mean(), inplace=True)

# Replace categorical missing values with the column mode
for column in df.select_dtypes(include=['object']):
    df[column].fillna(df[column].mode()[0], inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Step 5: Univariate Analysis
# Visualizing the distribution of scores
plt.figure(figsize=(10, 5))
sns.histplot(df["math_score"], kde=True, bins=20, color="blue", label="Math Score")
sns.histplot(df["reading_score"], kde=True, bins=20, color="green", label="Reading Score")
sns.histplot(df["writing_score"], kde=True, bins=20, color="red", label="Writing Score")
plt.legend()
plt.title("Distribution of Scores")
plt.show()

# Countplot for gender distribution
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="gender", palette="Set2")
plt.title("Gender Distribution")
plt.show()

# Step 6: Bivariate Analysis
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot of scores
sns.pairplot(df[["math_score", "reading_score", "writing_score"]], diag_kind="kde")
plt.show()

# Step 7: Group Analysis
# Group by gender and average scores
gender_scores = df.groupby("gender")[["math_score", "reading_score", "writing_score"]].mean()
print("\nAverage Scores by Gender:")
print(gender_scores)

# Visualizing average scores by gender
gender_scores.plot(kind="bar", figsize=(8, 5), color=["blue", "green", "red"])
plt.title("Average Scores by Gender")
plt.ylabel("Average Score")
plt.show()

# Step 8: Insights Summary
print("\nKey Insights:")
print("1. Check distribution of scores to identify performance trends.")
print("2. Analyze gender-based differences in scores.")
print("3. Investigate relationships between math, reading, and writing scores using correlation and pairplots.")
```


## Apply dimensionality reduction or feature selection techniques on a high-dimensional dataset. Explain the steps and justify the features retained.


```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# Step 1: Generate a synthetic high-dimensional dataset
# Creating a dataset with 1000 features, of which only 50 are informative
X, y = make_classification(n_samples=1000, n_features=1000, n_informative=50, 
                           n_redundant=950, random_state=42)

# Convert to a DataFrame for easier handling
data = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
labels = pd.Series(y, name="Target")

# Step 2: Preprocessing - Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply Principal Component Analysis (PCA)
# Keep components explaining 95% of the variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Number of components retained
print(f"Number of components retained by PCA: {pca.n_components_}")

# Step 4: Feature Selection using Random Forest
# Fit a Random Forest classifier to find important features
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_scaled, y)

# Select features with importance above a threshold
sfm = SelectFromModel(forest, threshold="mean")
sfm.fit(X_scaled, y)
selected_features = sfm.transform(X_scaled)

# Number of features retained
print(f"Number of features retained by feature selection: {selected_features.shape[1]}")

# Step 5: Compare retained features
# Plot explained variance ratio for PCA
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")
plt.grid()
plt.show()

# Feature importance from Random Forest
feature_importances = forest.feature_importances_
important_features = np.argsort(feature_importances)[-10:][::-1]

# Display the top 10 important features
print("Top 10 features selected by Random Forest:")
for idx in important_features:
    print(f"Feature_{idx+1}: Importance = {feature_importances[idx]:.4f}")

```

## Train a linear regression model on a real-estate dataset using scikit-learn. Evaluate its performance using R² and mean square error (MSE).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load a real-estate dataset (example: Boston housing dataset from sklearn)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)

# Prepare the data
df = data.frame
X = df.drop('MedHouseVal', axis=1)  # Features
y = df['MedHouseVal']              # Target (Median House Value)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

```

## Build a logistic regression model on a cancer diagnosis dataset. Explain the results and performance metrics like accuracy, precision, recall, and F1-score.

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the breast cancer dataset from scikit-learn
data = load_breast_cancer()

# Prepare the dataset
df = pd.DataFrame(data=data.data, columns=data.feature_names)
X = df  # Features
y = data.target  # Target variable (Malignant = 1, Benign = 0)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
```


