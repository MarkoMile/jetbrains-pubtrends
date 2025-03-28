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
import plotly.graph_objects as go
import gzip
import os
import ftplib
import tempfile
import time

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('GEO Dataset Clustering and Visualization', 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Label('Select Data Retrieval Mode:'),
        dcc.Dropdown(
            id='retrieval-mode',
            options=[
                {'label': 'Entrez - API Mode', 'value': 'entrez'},
                {'label': 'FTP - Full Data Mode (unstable)', 'value': 'ftp'}
            ],
            value='entrez',
            style={'width': '100%', 'marginBottom': 20}
        ),
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
          html.Div(id='timing-label', style={'textAlign': 'center', 'marginTop': 20}),
          html.Div(id='dbscan-plot-container'),
          html.Div(id='agglo-plot-container')
      ]
    ),
    html.Div(id='error-output', style={'color': 'red', 'marginBottom': 20}),
])

# Add a global variable to store the start time
start_time = None

def process_pmids_entrez(pmids_text):
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

def process_pmids_ftp(pmids_text):
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
        
        # Get metadata for GEO datasets using FTP mode
        metadata_list = []
        geo_accessions = {}
        
        # Fetch accession numbers for each GEO ID
        for geo_id in geo_ids:
            # Use efetch to get the accession number
            efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gds&id={geo_id}"
            efetch_response = requests.get(efetch_url)
            
            # Parse the response to extract the accession number
            for line in efetch_response.text.splitlines():
                if "Series" in line and "Accession:" in line:
                    # Extract the accession number (e.g., GSE216999)
                    parts = line.split()
                    accession_index = parts.index("Accession:") + 1
                    if accession_index < len(parts):
                        # Remove the "GSE" prefix from the accession number if it exists
                        accession = parts[accession_index]
                        if accession.startswith("GSE"):
                            accession = accession[3:]  # Remove the first 3 characters ("GSE")
                            parts[accession_index] = accession
                        geo_accessions[geo_id] = parts[accession_index]
                        break
        
        # Fetch metadata for each GEO dataset using FTP
        for geo_id, accession in geo_accessions.items():
            try:
                # Determine the FTP directory structure based on accession number
                if accession.isdigit():
                    range_dir = f"GSE{accession[:-3]}nnn"
                    # Parse FTP URL components
                    ftp_path = f"/geo/series/{range_dir}/GSE{accession}/soft/GSE{accession}_family.soft.gz"
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=True,delete_on_close=False,suffix='.gz') as temp_file:
                        temp_path = temp_file.name
                        
                        # Download file via FTP
                        ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
                        ftp.login()  # anonymous login
                        try:
                            ftp.retrbinary(f'RETR {ftp_path}', temp_file.write)
                        finally:
                            ftp.quit()
                            temp_file.close()

                        # Process the downloaded file
                        with gzip.open(temp_path, 'rt', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # Extract metadata from the SOFT file
                            title = ""
                            experiment_type = ""
                            summary = ""
                            organism = ""
                            overall_design = ""
                            
                            for line in content.splitlines():
                                if line.startswith("!Series_title"):
                                    title = line.split("=")[1].strip()
                                elif line.startswith("!Series_type"):
                                    experiment_type = line.split("=")[1].strip()
                                elif line.startswith("!Series_summary"):
                                    summary = line.split("=")[1].strip()
                                elif line.startswith("!Platform_organism"): # I am not sure if it should be sample or platform organism
                                    organism = line.split("=")[1].strip()
                                elif line.startswith("!Series_overall_design"):
                                    overall_design = line.split("=")[1].strip()
                            
                            # Combine metadata fields
                            metadata = f"{title} {experiment_type} {summary} {organism} {overall_design}"
                            metadata_list.append(metadata)
                
            except Exception as e:
                print(f"Error fetching metadata for GSE{accession}: {str(e)}")
            
        # Ensure we have metadata for each GEO ID
        if len(metadata_list) == 0:
            return None, "Failed to retrieve metadata for any GEO datasets"
        elif len(metadata_list) != len(geo_accessions.items()):
            return None, "Failed to retrieve metadata for some GEO datasets."
        
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
     Output('error-output', 'children'),
     Output('timing-label', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('pmids-input', 'value'),
     State('retrieval-mode', 'value')]
)
def update_graph(n_clicks, pmids_text, ret_mode):
    global start_time
    
    if n_clicks == 0:
        return None, None, None, None
        
    if not pmids_text:
        return None, None, "Please enter PMIDs", None

    # Record start time when button is clicked
    if start_time is None:
        start_time = time.time()

    if ret_mode == 'entrez':    
        plots, error = process_pmids_entrez(pmids_text)
    elif ret_mode == 'ftp':
        plots, error = process_pmids_ftp(pmids_text)
    else:
        plots, error = None, "unexpected error, check retrieval mode"
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    timing_text = f"Graphs generated in {elapsed_time:.2f} seconds"
    
    # Reset start time for next run
    start_time = None
    
    if error:
        return None, None, error, None
        
    return plots[0], plots[1], None, timing_text

if __name__ == '__main__':
    app.run(debug=False) 