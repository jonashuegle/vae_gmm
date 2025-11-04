import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
from matplotlib import pyplot as plt

def visualize_tsne(latent_features, 
                   cluster_assignments, 
                   timestamps = None, 
                   n_components=3, 
                   colors = None,  
                   perplexity=30, 
                   max_iter=300, 
                   random_state=42,
                   cluster_names = None,
                   height=600, 
                   width=600):
    if n_components not in [2, 3]:
        raise ValueError("n_components muss 2 oder 3 sein")
    
    if colors is not None:
        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in colors]

    # t-SNE Transformation
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, max_iter=max_iter)
    tsne_results = tsne.fit_transform(latent_features)

    # Erstellen Sie ein DataFrame mit den Ergebnissen
    if timestamps is not None:
        data = pd.DataFrame({
            'Feature 1': tsne_results[:, 0],
            'Feature 2': tsne_results[:, 1],
            'Cluster': cluster_assignments,
            'Date': pd.to_datetime(timestamps)
        })

    else:
        data = pd.DataFrame({
            'Feature 1': tsne_results[:, 0],
            'Feature 2': tsne_results[:, 1],
            'Cluster': cluster_assignments,
            'Date': pd.to_datetime(pd.Timestamp('NaT'))
        })

    if n_components == 3:
        data['Feature 3'] = tsne_results[:, 2]

    # Erstellen Sie einen Scatter Plot
    if n_components == 2:
        fig = go.Figure()
    else:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Sortieren Sie die Cluster-IDs
    sorted_clusters = sorted(data['Cluster'].unique())

    for i, cluster in enumerate(sorted_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        # Name auswählen:
        if cluster_names is not None:
            cluster_label = cluster_names[i]
        else:
            cluster_label = f'Cluster {cluster+1}'
        
        if n_components == 2:
            trace = go.Scatter(
                x=cluster_data['Feature 1'],
                y=cluster_data['Feature 2'],
                mode='markers',
                marker=dict(size=5, 
                            opacity=1,
                            color=colors[i] if colors is not None else None),
                name=cluster_label,   # <--- Regime-Name oder Default
                text=cluster_data['Date'].dt.strftime('%Y-%m-%d'),
                hoverinfo='name+text',
                legendgroup=cluster_label,
                showlegend=True,
                marker_symbol='circle',
                marker_size=5,
            )
        else:
            trace = go.Scatter3d(
                x=cluster_data['Feature 1'],
                y=cluster_data['Feature 2'],
                z=cluster_data['Feature 3'],
                mode='markers',
                marker=dict(size=5, 
                            opacity=1,
                            color=colors[i] if colors is not None else None),
                name=cluster_label,
                text=cluster_data['Date'].dt.strftime('%Y-%m-%d'),
                hoverinfo='name+text',
                legendgroup=cluster_label,
                showlegend=True,
                marker_symbol='circle',
                marker_size=5,
            )
        fig.add_trace(trace)

    # Update Layout
    fig.update_layout(
        title=f't-SNE {n_components}D Visualization of the Latent Space',
        height=height,
        width=width,
        scene=dict(
            xaxis_title='t-SNE Feature 1',
            yaxis_title='t-SNE Feature 2',
            zaxis_title='t-SNE Feature 3' if n_components == 3 else None
        ) if n_components == 3 else None,
        xaxis_title='t-SNE Feature 1' if n_components == 2 else None,
        yaxis_title='t-SNE Feature 2' if n_components == 2 else None,
        legend=dict(
            font=dict(size=14),  # Erhöht die Schriftgröße in der Legende
            itemsizing='constant'  # Dies sollte die Größe der Legendeneinträge konsistent halten
        )
    )

    return fig, ax


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_tsne_matplotlib(
    latent_features, 
    cluster_probabilities,         # 1D Labels ODER 2D Wahrscheinlichkeiten!
    cluster_names=None,            # Liste der Namen (z.B. ["NAO+",...])
    cluster_colors=None,           # Liste der RGB-Farben (gleiche Reihenfolge wie cluster_names)
    perplexity=30, 
    random_state=42, 
    fig=None,
    ax=None,
    figsize=(5,5),
    legend_kwargs=None,
    s=10,
    **kwargs
):
    # --- t-SNE Dimensionality Reduction ---
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(latent_features)
    
    # --- Clusterlabels und Transparenz ---
    cluster_probabilities = np.asarray(cluster_probabilities)
    if cluster_probabilities.ndim == 2:
        cluster_assignments = np.argmax(cluster_probabilities, axis=1)
        alpha_values = np.max(cluster_probabilities, axis=1)
        n_clusters = cluster_probabilities.shape[1]
    elif cluster_probabilities.ndim == 1:
        cluster_assignments = np.array(cluster_probabilities)
        alpha_values = np.ones_like(cluster_assignments, dtype=float)
        n_clusters = len(np.unique(cluster_assignments))
    else:
        raise ValueError("cluster_probabilities must be 1D (labels) or 2D (probabilities) array.")
    
    # --- Farben und Labels ---
    unique_clusters = sorted(np.unique(cluster_assignments))
    if cluster_names is not None:
        # Sortiere die Namen passend zu den tatsächlichen Indizes!
        legend_labels = [cluster_names[c] for c in unique_clusters]
        if cluster_colors is not None:
            colors = [cluster_colors[c] for c in unique_clusters]
        else:
            colors = sns.color_palette("Set1", n_colors=len(unique_clusters))
    else:
        legend_labels = [str(c) for c in unique_clusters]
        colors = sns.color_palette("Set1", n_colors=len(unique_clusters))
    color_map = {c: color for c, color in zip(unique_clusters, colors)}
    
    # --- Plot ---
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    colors_arr = np.array([color_map[c] for c in cluster_assignments])
    sc = ax.scatter(
        tsne_results[:, 0], tsne_results[:, 1],
        c=colors_arr, alpha=alpha_values, s=s,
        edgecolor='k', linewidths=0.3, **kwargs
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    # --- Legende ---
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[c],
                   markersize=8, label=legend_labels[i], markeredgecolor='k', markeredgewidth=0.5)
        for i, c in enumerate(unique_clusters)
    ]
    legend_args = dict(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.22),
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=10
    )
    if legend_kwargs is not None:
        legend_args.update(legend_kwargs)
    ax.legend(**legend_args)
    plt.tight_layout()
    return fig, ax


