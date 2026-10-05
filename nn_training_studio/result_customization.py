"""Bounded, aligned training feature samples and local plot helpers."""

import numpy as np
import pandas as pd


def make_feature_sample(values, actual=None, predicted=None, class_names=None,
                        max_samples=1000, max_features=128):
    """Retain aligned held-out rows; flatten windows and cap rows/coordinates."""
    values = np.asarray(values)
    if values.ndim < 2 or not len(values):
        return pd.DataFrame()
    rows = np.sort(np.random.default_rng(42).choice(
        len(values), min(len(values), max_samples), replace=False))
    flat = values[rows].reshape(len(rows), -1)
    columns = np.linspace(0, flat.shape[1] - 1,
                          min(flat.shape[1], max_features), dtype=int)
    table = pd.DataFrame(flat[:, columns],
                         columns=[f"feature_{index + 1}" for index in columns])
    table.insert(0, "record_number", rows + 1)
    names = list(class_names or [])
    for name, labels in (("actual_label", actual), ("predicted_label", predicted)):
        if labels is None:
            continue
        labels = np.asarray(labels)
        if labels.size != len(values):
            continue
        selected = labels.reshape(-1)[rows]
        table[name] = [names[int(value)] if names and 0 <= int(value) < len(names)
                       else str(value) for value in selected]
    return table


def compute_tsne(frame, columns=None, perplexity=30, max_samples=1000):
    """Return reproducible local t-SNE coordinates with original row indices.

    Labels and record numbers are never used as features. Training tables use
    feature_* columns. For other numeric tables, supply columns explicitly.
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    if columns is None:
        columns = [str(c) for c in frame.columns if str(c).startswith("feature_")]
    forbidden = {"record_number", "actual_label", "predicted_label", "actual_class_index",
                 "predicted_class_index", "correct", "epoch"}
    if not columns or forbidden.intersection(columns):
        raise ValueError("Choose numeric feature columns, excluding labels and record numbers.")
    if not 3 <= int(max_samples) <= 1000:
        raise ValueError("t-SNE sample limit must be between 3 and 1000.")
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    numeric = numeric.loc[:, numeric.notna().any()]
    if len(numeric) < 3 or numeric.shape[1] == 0:
        raise ValueError("t-SNE needs at least three rows with numeric features.")
    numeric = numeric.fillna(numeric.median())
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] == 0:
        raise ValueError("t-SNE needs at least one varying feature.")
    if len(numeric) > max_samples:
        numeric = numeric.sample(n=int(max_samples), random_state=42).sort_index()
    perplexity = float(perplexity)
    if not np.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("t-SNE perplexity must be a finite positive number.")
    scaled = StandardScaler().fit_transform(numeric)
    # Bound wide user-selected tables before the pairwise embedding operation.
    if scaled.shape[1] > 50:
        from sklearn.decomposition import PCA
        scaled = PCA(n_components=min(50, len(scaled) - 1), random_state=42).fit_transform(scaled)
    coordinates = TSNE(n_components=2, perplexity=min(perplexity, len(scaled) - 1),
                       init="random", learning_rate="auto", random_state=42).fit_transform(scaled)
    return pd.DataFrame(coordinates, index=numeric.index, columns=["tsne_1", "tsne_2"])


def extra_plot_code(plot_type, table_name, frame):
    """Editable templates for training curves, heatmaps, and t-SNE."""
    prefix = ("def create_plot(context):\n"
              "    plt = context['plt']\n"
              "    np = context['np']\n"
              f"    df = context['tables'][{table_name!r}]\n")
    if plot_type in ("Loss & Accuracy", "Training Curves"):
        if "loss" not in frame:
            raise ValueError("Choose the training_history table for training curves.")
        body = '''    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = df['epoch'] if 'epoch' in df else np.arange(1, len(df) + 1)
    for column in ['loss', 'val_loss']:
        if column in df:
            axes[0].plot(epochs, df[column], label=column)
    for column in df.columns:
        if column not in ['epoch', 'loss', 'val_loss', 'learning_rate', 'lr']:
            axes[1].plot(epochs, df[column], label=column)
    for ax, title in zip(axes, ['Training / validation loss', 'Training / validation metrics']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        if ax.lines:
            ax.legend()
        ax.grid(True, alpha=0.25)
'''
    elif plot_type == "Correlation Heatmap":
        columns = [str(c) for c in frame.select_dtypes(include=[np.number]).columns
                   if str(c) not in ('record_number', 'actual_class_index', 'predicted_class_index')][:40]
        if len(columns) < 2:
            raise ValueError("A correlation heatmap needs at least two numeric columns.")
        body = (f"    matrix = df[{columns!r}].corr()\n" + '''    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix.to_numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title('Feature correlation heatmap (up to 40 columns)')
    fig.colorbar(image, ax=ax, label='Pearson correlation')
''')
    elif plot_type == "t-SNE":
        if not any(str(c).startswith('feature_') for c in frame.columns):
            raise ValueError("Choose embedding_features from a completed training run for t-SNE.")
        body = '''    points = context['compute_tsne'](df, perplexity=30, max_samples=1000)
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = df.loc[points.index, 'actual_label'].astype(str) if 'actual_label' in df else None
    if labels is None:
        ax.scatter(points['tsne_1'], points['tsne_2'], s=20, alpha=0.75)
    else:
        for label in sorted(labels.unique()):
            group = points.loc[labels == label]
            ax.scatter(group['tsne_1'], group['tsne_2'], s=20, alpha=0.75, label=label)
        ax.legend(title='Actual class')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title('t-SNE of held-out features (sampled; exploratory)')
'''
    else:
        raise ValueError(f"Unsupported template: {plot_type}")
    return prefix + body + "    fig.tight_layout()\n    return fig\n"
