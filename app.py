"""
GEO Dataset Clustering and Visualization Web Service

This Dash web service performs clustering analysis on Gene Expression Omnibus (GEO) datasets
based on their metadata similarity. It takes a list of PubMed IDs as input and returns
an interactive visualization of the clustering results.

The workflow includes:
1. Accepting PMIDs from a web form
2. Fetching associated GEO datasets using NCBI E-utilities
3. Retrieving metadata for each GEO dataset
4. Converting metadata to TF-IDF vectors
5. Performing DBSCAN and Agglomerative clustering
6. Creating interactive visualizations with PCA dimensionality reduction

Technical notes:
    - Initially, the PMIDs are read from the form, and the geo_ids are retrieved using the NCBI E-utilities in bulk.
    - Then, the PMIDs are retrieved for each geo_id using the NCBI E-utilities element-wise, this is slow, because it requires a lot of requests.
    - Only PMIDs that are present in the form are considered - this is optional, and can be removed.
    - Then, the metadata is retrieved for each geo_id using the NCBI E-utilities.
    - Both DBSCAN and Agglomerative clustering are used to cluster the metadata vectors.
"""

import dash
from dash import html, dcc, Input, Output, State
import requests
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import numpy as np
import plotly.graph_objects as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('GEO Dataset Clustering and Visualization', 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Label('Enter PubMed IDs (one per line):'),
        dcc.Textarea(
            id='pmids-input',
            value='',
            style={'width': '100%', 'height': 200},
            placeholder='Enter PMIDs here, one per line...'
        ),
        html.Button('Generate Clusters', 
                   id='submit-button', 
                   n_clicks=0,
                   style={'marginTop': 10, 'marginBottom': 20})
    ], style={'width': '80%', 'margin': '0 auto'}),
    
    dcc.Loading(
      id="loading",
      type="default",
      children=[
          html.Div(id='dbscan-plot-container'),
          html.Div(id='agglo-plot-container')
      ]
    ),
    html.Div(id='error-output', style={'color': 'red', 'marginBottom': 20}),
])

def process_pmids(pmids_text):
    """Process the input PMIDs and generate the clustering visualization."""
    # Split the input text into PMIDs and clean them
    PMIDs_list = [pmid.strip() for pmid in pmids_text.split('\n') if pmid.strip()]
    
    if not PMIDs_list:
        return None, "Please enter at least one PMID"
    
    try:
        # Create a string of all PMIDs, separated by commas
        pmid_string = ",".join(PMIDs_list)
        
        # Fetch GEO IDs
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={pmid_string}&retmode=json"
        response = requests.get(url)
        root = response.json()
        
        if 'linksets' not in root or not root['linksets']:
            return None, "No GEO datasets found for the provided PMIDs"
            
        geo_ids = root['linksets'][0]['linksetdbs'][0]['links']
        if not geo_ids:
            return None, "No GEO datasets found for the provided PMIDs"
            
        # Create a dictionary to store the PMIDs for each geo_id
        pmids_dict = {}
        pmids_set = set(PMIDs_list)
        
        # Get PMIDs for each GEO ID
        for geo_id in geo_ids:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gds&db=pubmed&linkname=gds_pubmed&id={geo_id}&retmode=json"
            response = requests.get(url)
            root = response.json()
            pmids = set(root['linksets'][0]['linksetdbs'][0]['links'])
            pmids = pmids.intersection(pmids_set)
            pmids_dict[geo_id] = list(pmids)
        
        # Get metadata for GEO datasets
        geo_ids_string = ",".join(geo_ids)
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id={geo_ids_string}"
        response = requests.get(url)
        
        metadata_list = []
        root = ET.fromstring(response.text)
        for docsum in root.findall('.//DocSum'):
            geo_id = docsum.find('Id').text
            title = docsum.find('Item[@Name="title"]').text
            experiment_type = docsum.find('Item[@Name="gdsType"]').text
            summary = docsum.find('Item[@Name="summary"]').text
            organism = docsum.find('Item[@Name="taxon"]').text
            metadata_list.append(f"{title} {experiment_type} {summary} {organism}")
        
        # Vectorize metadata
        vectorizer = TfidfVectorizer()
        vectorizer.fit(metadata_list)
        metadata_vectors = vectorizer.transform(metadata_list)
        
        # Perform clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan.fit(metadata_vectors)
        
        # Perform Agglomerative clustering
        agglo = AgglomerativeClustering(n_clusters=5)
        agglo_labels = agglo.fit_predict(metadata_vectors.toarray())
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        metadata_2d = pca.fit_transform(metadata_vectors.toarray())
        
        # Create hover text
        hover_texts = []
        for i, geo_id in enumerate(geo_ids):
            pmids = pmids_dict[geo_id]
            hover_text = f"GEO ID: {geo_id}<br>Related PMIDs:<br>" + "<br>".join(pmids)
            hover_texts.append(hover_text)
        
        # Create the DBSCAN plot
        dbscan_fig = go.Figure()
        dbscan_fig.add_trace(go.Scatter(
            x=metadata_2d[:, 0],
            y=metadata_2d[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=dbscan.labels_,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Cluster')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='GEO Datasets'
        ))
        
        dbscan_fig.update_layout(
            title='DBSCAN Clustering of GEO Metadata',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            showlegend=True,
            hovermode='closest'
        )
        
        # Create the Agglomerative clustering plot
        agglo_fig = go.Figure()
        agglo_fig.add_trace(go.Scatter(
            x=metadata_2d[:, 0],
            y=metadata_2d[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=agglo_labels,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Cluster')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='GEO Datasets'
        ))
        
        agglo_fig.update_layout(
            title='Agglomerative Clustering of GEO Metadata',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            showlegend=True,
            hovermode='closest'
        )
        
        return [dcc.Graph(id='dbscan-plot', figure=dbscan_fig),
                dcc.Graph(id='agglo-plot', figure=agglo_fig)], None
        
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

@app.callback(
    [Output('dbscan-plot-container', 'children'),
     Output('agglo-plot-container', 'children'),
     Output('error-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('pmids-input', 'value')]
)
def update_graph(n_clicks, pmids_text):
    
    if n_clicks == 0:
        return None, None, None
        
    if not pmids_text:
        return None, None, "Please enter PMIDs"
        
    plots, error = process_pmids(pmids_text)
    if error:
        return None, None, error
        
    return plots[0], plots[1], None

if __name__ == '__main__':
    app.run(debug=False) 