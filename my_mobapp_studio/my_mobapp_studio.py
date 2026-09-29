import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix


# ================================
# 1. Load dataset
# ================================
def load_dataset(path="googleplaystore.csv"):
    return pd.read_csv(path)


# ================================
# 2. Dataset summary
# ================================
def print_summarize_dataset(dataset):
    print("=== SHAPE ===")
    print(dataset.shape)
    print("\n=== HEAD ===")
    print(dataset.head())
    print("\n=== INFO ===")
    print(dataset.info())
    print("\n=== DESCRIBE ===")
    print(dataset.describe(include="all"))


# ================================
# 3. Cleaning
# ================================
def clean_dataset(dataset):
    df = dataset.copy()

    df = df.drop_duplicates()
    df = df.dropna(subset=["Rating"])

    df["Installs"] = df["Installs"].str.replace("+", "", regex=False)
    df["Installs"] = df["Installs"].str.replace(",", "", regex=False)
    df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")
    df = df.dropna(subset=["Installs"])

    df["Price"] = df["Price"].str.replace("$", "", regex=False)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    df = df.dropna(subset=["Reviews"])

    return df


# ================================
# 4. Histograms
# ================================
def print_histograms(dataset):
    numeric_cols = dataset.select_dtypes(include=np.number).columns
    dataset[numeric_cols].hist(figsize=(12, 10), bins=20)
    plt.tight_layout()
    plt.show()


# ================================
# 5. Correlation matrix
# ================================
def compute_correlations_matrix(dataset):
    corr = dataset.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Matrice des corrélations")
    plt.show()
    return corr


# ================================
# 6. Scatter matrix
# ================================
def print_scatter_matrix(dataset):
    numeric_cols = dataset.select_dtypes(include=np.number)
    scatter_matrix(numeric_cols, figsize=(12, 10), diagonal="kde")
    plt.show()


# ================================
# 7. Paid Family – Popular apps bar chart
# ================================
def plot_paid_family_popular(dataset):
    df = dataset[(dataset["Category"] == "FAMILY") & (dataset["Price"] > 0)]
    df = df.sort_values("Installs", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(df["App"], df["Installs"])
    plt.gca().invert_yaxis()
    plt.title("Top apps payantes de la catégorie FAMILY")
    plt.xlabel("Installations")
    plt.show()


# ================================
# 8. Genres popular in paid Family apps
# ================================
def plot_genres_paid_family(dataset):
    df = dataset[(dataset["Category"] == "FAMILY") & (dataset["Price"] > 0)]
    counts = df["Genres"].value_counts().head(10)

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title("Genres populaires – Apps payantes Family")
    plt.show()


# ================================
# 9. Installations per category (table)
# ================================
def table_installations_per_category(dataset):
    table = dataset.groupby("Category")["Installs"].sum().sort_values(ascending=False)
    print(table)
    return table


# ================================
# 10. Pie chart installations per category
# ================================
def plot_pie_installations_per_category(dataset):
    table = dataset.groupby("Category")["Installs"].sum()
    plt.figure(figsize=(10, 10))
    plt.pie(table, labels=table.index, autopct="%1.1f%%")
    plt.title("Installations par catégorie")
    plt.show()


# ================================
# 11. Mean price per category
# ================================
def plot_mean_price_per_category(dataset):
    prices = dataset.groupby("Category")["Price"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    prices.plot(kind="bar")
    plt.title("Prix moyen par catégorie")
    plt.ylabel("Prix moyen ($)")
    plt.show()


# ================================
# 12. Most expensive apps per category
# ================================
def most_expensive_apps(dataset):
    res = dataset.loc[dataset.groupby("Category")["Price"].idxmax()][["Category", "App", "Price"]]
    print(res)
    return res


# ================================
# 13. MAIN EXECUTION PIPELINE
# ================================
if __name__ == "__main__":

    dataset = load_dataset("googleplaystore.csv")

    print_summarize_dataset(dataset)

    clean_df = clean_dataset(dataset)

    print_histograms(clean_df)

    compute_correlations_matrix(clean_df)

    print_scatter_matrix(clean_df)

    plot_paid_family_popular(clean_df)

    plot_genres_paid_family(clean_df)

    table_installations_per_category(clean_df)

    plot_pie_installations_per_category(clean_df)

    plot_mean_price_per_category(clean_df)

    most_expensive_apps(clean_df)
