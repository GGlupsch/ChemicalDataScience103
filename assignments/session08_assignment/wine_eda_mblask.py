"""Exploratory data analysis and classification of the Wine dataset."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Figures are saved to files.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def load_data():
    """Load the Wine dataset into a DataFrame."""
    wine = load_wine()

    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target

    return df, wine.feature_names


def inspect_data(df, feature_names):
    """Print an overview of the dataset and check its quality."""
    print("\n--- 1. Dataset overview ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nFirst 10 observations:")
    print(df.head(10).to_string(index=False))

    print("\nDescriptive statistics:")
    print(df[feature_names].describe().round(2).to_string())

    print("\n--- 2. Basic exploration ---")
    print("\nNumber of observations per class:")
    print(df["target"].value_counts().sort_index())

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nNumber of duplicate rows:", df.duplicated().sum())

    # Look at each pair of features once, ignoring self-correlations.
    correlation = df[feature_names].corr()
    upper_mask = np.triu(np.ones(correlation.shape), k=1).astype(bool)
    feature_pairs = correlation.where(upper_mask).stack()

    sorted_pairs = feature_pairs.abs().sort_values(ascending=False)
    strongest_pairs = feature_pairs.loc[sorted_pairs.index[:5]]

    print("\nFive strongest feature correlations:")
    print(strongest_pairs.round(3))

    selected_features = [
        "alcohol",
        "flavanoids",
        "color_intensity",
        "od280/od315_of_diluted_wines",
        "proline",
    ]

    print("\nMean values of selected features by class:")
    print(df.groupby("target")[selected_features].mean().round(2))


def save_visualisations(df, feature_names):
    """Create and save three plots for exploratory analysis."""
    FIGURE_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    labelled = df.copy()
    labelled["wine_class"] = "Class " + labelled["target"].astype(str)

    # Pairplot of five selected chemical properties.
    selected_features = [
        "alcohol",
        "malic_acid",
        "flavanoids",
        "color_intensity",
        "proline",
    ]

    plot = sns.pairplot(
        data=labelled,
        vars=selected_features,
        hue="wine_class",
        corner=True,
        diag_kind="hist",
        height=2.0,
        plot_kws={"alpha": 0.7, "s": 21, "edgecolor": "none"},
        diag_kws={"alpha": 0.55},
    )
    plot.fig.suptitle("Wine dataset: selected chemical features", y=1.02)
    plot.savefig(FIGURE_DIR / "pairplot.png", dpi=180, bbox_inches="tight")
    plt.close(plot.fig)

    # Compare flavanoid levels between classes.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=labelled, x="wine_class", y="flavanoids", ax=ax)

    ax.set_title("Flavanoid content by wine class")
    ax.set_xlabel("Wine class")
    ax.set_ylabel("Flavanoids")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "boxplot_flavanoids.png", dpi=180)
    plt.close(fig)

    # Correlations between the 13 features (without the target labels).
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = df[feature_names].corr()

    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )

    ax.set_title("Correlations between wine features")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def explore_pca(df, feature_names):
    """Standardise the features and visualise the first two PCs."""
    print("\n--- 4. Principal component analysis ---")

    X = df[feature_names]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled_df)

    pca_df = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
    pca_df["wine_class"] = "Class " + y.astype(str).to_numpy()

    variance = pca.explained_variance_ratio_ * 100
    print(f"PC1 explained variance: {variance[0]:.2f}%")
    print(f"PC2 explained variance: {variance[1]:.2f}%")
    print(f"Combined explained variance: {variance.sum():.2f}%")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PC1", "PC2"],
    )

    for component in ["PC1", "PC2"]:
        top_features = loadings[component].abs().nlargest(5).index
        print(f"\nFive largest loadings for {component}:")
        print(loadings.loc[top_features, component].round(3))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="wine_class",
        s=55,
        alpha=0.85,
        ax=ax,
    )

    ax.set_title("Wine samples projected onto the first two PCs")
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% explained variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% explained variance)")
    ax.legend(title="Wine class")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pca_scatter.png", dpi=180)
    plt.close(fig)

    return X, y


def compare_models(X, y):
    """Compare classifiers using all features and two PCA components."""
    print("\n--- 5. Classification ---")

    # Use the same training and test observations in every comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit preprocessing steps on the training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    models = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Decision tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    results = []

    for name, model in models.items():
        # Train with all 13 standardised features.
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy_full = accuracy_score(y_test, predictions)

        # Tree models can also tell us which original features they used.
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=X.columns)
            importance = importance.sort_values(ascending=False)

            print(f"\n{name}: five most important features")
            print(importance.head(5).round(3))

        # Repeat the classification using only the two PCA components.
        model_pca = clone(model)
        model_pca.fit(X_train_pca, y_train)
        predictions_pca = model_pca.predict(X_test_pca)
        accuracy_pca = accuracy_score(y_test, predictions_pca)

        results.append({
            "Model": name,
            "Accuracy (13 features)": accuracy_full,
            "Accuracy (2 PCs)": accuracy_pca,
            "Correct (13 features)": sum(predictions == y_test),
            "Correct (2 PCs)": sum(predictions_pca == y_test),
        })

    results_df = pd.DataFrame(results).set_index("Model")

    print("\nClassification results:")
    print(results_df.round(3).to_string())
    print("\nLogistic regression is used because the target contains three classes.")


def main():
    """Run all parts of the analysis."""
    df, feature_names = load_data()

    inspect_data(df, feature_names)
    save_visualisations(df, feature_names)
    X, y = explore_pca(df, feature_names)
    compare_models(X, y)

    print(f"\nPlots saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
