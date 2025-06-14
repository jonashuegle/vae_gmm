import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def visualize_tsne(latent_features, cluster_assignments, timestamps = None, n_components=3, colors = None,  perplexity=30, max_iter=300, random_state=42, height = 600, width = 600):
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
        if n_components == 2:
            trace = go.Scatter(
                x=cluster_data['Feature 1'],
                y=cluster_data['Feature 2'],
                mode='markers',
                marker=dict(size=5, 
                            opacity=1,
                            color=colors[i]),
                name=f'Cluster {cluster+1}',
                text=cluster_data['Date'].dt.strftime('%Y-%m-%d'),
                hoverinfo='name+text',
                legendgroup=f'Cluster {cluster+1}',
                showlegend=True,
                marker_symbol='circle',
                marker_size=5,  # Größere Marker in der Legende
            )
        else:
            trace = go.Scatter3d(
                x=cluster_data['Feature 1'],
                y=cluster_data['Feature 2'],
                z=cluster_data['Feature 3'],
                mode='markers',
                marker=dict(size=5, 
                            opacity=1,
                            color=colors[i]),
                name=f'Cluster {cluster+1}',
                text=cluster_data['Date'].dt.strftime('%Y-%m-%d'),
                hoverinfo='name+text',
                legendgroup=f'Cluster {cluster+1}',
                showlegend=True,
                marker_symbol='circle',
                marker_size=5,  # Größere Marker in der Legende
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

    return fig



# Beispielaufruf:
# fig_2d = visualize_tsne(latent_features, cluster_assignments, timestamps, n_components=2)
# fig_2d.show()

# fig_3d = visualize_tsne(latent_features, cluster_assignments, timestamps, n_components=3)
# fig_3d.show()