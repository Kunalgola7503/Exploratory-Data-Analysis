# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns for better visibility
pd.set_option('display.max_columns', None)

# Setting seaborn style for clean visualizations
sns.set_style("whitegrid")

# --------------------------------------------
# Load Dataset
# --------------------------------------------
df2 = pd.read_csv(r"C:\Users\user\Downloads\House_Price.csv")

# Preview first few rows
df2.head()

# --------------------------------------------
# Basic Dataset Information
# --------------------------------------------
print("Shape:", df2.shape)  # Rows and columns
print("\nInfo:")
print(df2.info())  # Data types, non-null counts
print("\nSummary Statistics:")
print(df2.describe(include="all"))  # Statistical overview of all columns

# --------------------------------------------
# Missing Values & Duplicate Check
# --------------------------------------------
print("Missing Values:\n", df2.isnull().sum())
print("\nMissing %:\n", (df2.isnull().mean() * 100).round(2))
print("\nDuplicate Rows:", df2.duplicated().sum())

# --------------------------------------------
# Numerical Feature Distributions
# --------------------------------------------
df2.hist(figsize=(12, 8), bins=30)
plt.suptitle("Numerical Feature Distributions", fontsize=14)
plt.show()

# --------------------------------------------
# Categorical Feature Distributions
# --------------------------------------------
for col in df2.select_dtypes(include=["object"]).columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df2, palette="Set2")
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# --------------------------------------------
# Correlation Heatmap (Numerical Features Only)
# --------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# Boxplots: Numerical Features by Categorical Features
# --------------------------------------------
categorical_cols = df2.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    for num_col in df2.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=col, y=num_col, data=df2, palette="Set3")
        plt.title(f"{num_col} by {col}")
        plt.xticks(rotation=45)
        plt.show()

# --------------------------------------------
# Outlier Detection via Boxplots (Numerical Only)
# --------------------------------------------
for col in df2.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df2[col], color="orange")
    plt.title(f"Boxplot of {col}")
    plt.show()

# --------------------------------------------
# Final Insights Summary
# --------------------------------------------
print(f"Final EDA Insights")
print(f"1. Dataset contains {df2.shape[0]} rows and {df2.shape[1]} columns.")
print(f"2. Missing values detected in {df2.isnull().any().sum()} columns.")
print(f"3. Duplicate rows: {df2.duplicated().sum()}")
print(f"4. Numerical features show varied distributions (see histograms).")
print(f"5. Some categorical features appear imbalanced (see count plots).")
print(f"6. Correlation heatmap highlights key relationships between numeric features.")
print(f"7. Outliers identified in select numerical columns via boxplots.")
