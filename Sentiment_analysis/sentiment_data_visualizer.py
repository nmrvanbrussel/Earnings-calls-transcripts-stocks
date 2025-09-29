import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "CSV",
    "Sentiment_analysis_Data",
    "finbert_embeddings_plus_probs.parquet",
)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "figs")


def load_features(parquet_path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load parquet and infer probability and embedding columns."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Input parquet not found: {parquet_path}. Run vectorizing_data.py first."
        )
    df = pd.read_parquet(parquet_path)

    probability_columns = [c for c in df.columns if c.startswith("finbert_prob_")]
    embedding_columns = [c for c in df.columns if c.startswith("finbert_emb_")]

    if len(probability_columns) != 3:
        raise ValueError(
            f"Expected 3 probability columns, found {len(probability_columns)}: {probability_columns}"
        )
    if len(embedding_columns) == 0:
        raise ValueError("No embedding columns (finbert_emb_*) found.")
    if "finbert_pred_label" not in df.columns or "finbert_confidence" not in df.columns:
        raise ValueError("Missing finbert_pred_label and/or finbert_confidence columns.")
    return df, probability_columns, embedding_columns


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def plot_label_distribution(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(6, 4))
    order = sorted(df["finbert_pred_label"].astype(str).unique())
    ax = sns.countplot(x="finbert_pred_label", data=df, order=order)
    ax.set_title("FinBERT Predicted Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=150)
    plt.close()


def plot_confidence_histogram(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.histplot(df["finbert_confidence"], bins=30, kde=True)
    ax.set_title("FinBERT Confidence (Max Class Probability)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_hist.png"), dpi=150)
    plt.close()


def plot_probability_distributions(df: pd.DataFrame, probability_columns: list[str], output_dir: str) -> None:
    plt.figure(figsize=(7, 4))
    for col in probability_columns:
        sns.kdeplot(df[col], fill=True, label=col, alpha=0.35)
    plt.title("Class Probability Distributions")
    plt.xlabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probability_kdes.png"), dpi=150)
    plt.close()


def plot_pca_scatter(df: pd.DataFrame, embedding_columns: list[str], output_dir: str) -> None:
    features_matrix = df[embedding_columns].values
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(features_matrix)
    pca_df = pd.DataFrame(
        {
            "pca1": z[:, 0],
            "pca2": z[:, 1],
            "label": df["finbert_pred_label"].astype(str),
            "confidence": df["finbert_confidence"],
        }
    )
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(
        data=pca_df,
        x="pca1",
        y="pca2",
        hue="label",
        size="confidence",
        alpha=0.75,
        sizes=(20, 140),
    )
    explained = float(np.sum(pca.explained_variance_ratio_[:2]))
    ax.set_title(f"PCA of FinBERT Embeddings (explained ~{explained:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter.png"), dpi=150)
    plt.close()


def plot_pca_interactive(df: pd.DataFrame, embedding_columns: list[str], output_dir: str) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        # Plotly not installed; skip interactive plot silently
        return

    features_matrix = df[embedding_columns].values
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(features_matrix)
    pca_df = pd.DataFrame(
        {
            "pca1": z[:, 0],
            "pca2": z[:, 1],
            "label": df["finbert_pred_label"].astype(str),
            "confidence": df["finbert_confidence"],
            "Text": df["Text"] if "Text" in df.columns else "",
        }
    )
    fig = px.scatter(
        pca_df,
        x="pca1",
        y="pca2",
        color="label",
        size="confidence",
        hover_data=["Text", "confidence"],
        title="PCA of FinBERT Embeddings (interactive)",
    )
    fig.write_html(os.path.join(output_dir, "pca_scatter_interactive.html"))


def export_top_examples(df: pd.DataFrame, output_dir: str, top_n: int = 20) -> dict[str, str]:
    outputs: dict[str, str] = {}
    labels = sorted(df["finbert_pred_label"].astype(str).unique())
    for label in labels:
        subset = (
            df[df["finbert_pred_label"].astype(str) == label]
            .sort_values("finbert_confidence", ascending=False)
            .head(top_n)
        )
        keep = [c for c in ["finbert_pred_label", "finbert_confidence", "Text"] if c in subset.columns]
        out_path = os.path.join(output_dir, f"top_{label.lower()}_examples.csv")
        subset[keep].to_csv(out_path, index=False)
        outputs[label] = out_path
    return outputs


def main(parquet_path: str = DEFAULT_INPUT_PATH, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    ensure_output_dir(output_dir)
    df, prob_cols, emb_cols = load_features(parquet_path)

    plot_label_distribution(df, output_dir)
    plot_confidence_histogram(df, output_dir)
    plot_probability_distributions(df, prob_cols, output_dir)
    plot_pca_scatter(df, emb_cols, output_dir)
    plot_pca_interactive(df, emb_cols, output_dir)
    exported = export_top_examples(df, output_dir, top_n=20)

    print(f"Saved figures to: {output_dir}")
    print(f"Exported example tables: {exported}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    main(in_path, out_dir)


