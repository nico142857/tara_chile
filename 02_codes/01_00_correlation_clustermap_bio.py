#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram


# In[2]:


# CLR implementation
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.

    Parameters:
    data (pandas.DataFrame): A DataFrame with samples as rows and components as columns.

    Returns:
    pandas.DataFrame: A clr-normalized DataFrame.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for clr normalization.")

    # Add small amount to cells with a value of 0
    if (data <= 0).any().any():
        data = data.replace(0, eps)

    # Calculate the geometric mean of each row
    gm = np.exp(data.apply(np.log).mean(axis=1))

    # Perform clr transformation
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)

    return clr_data


# In[3]:


dir_omics = '../01_data/02_omics/'
dir_env = '../01_data/00_metadata/'

dir_out_correlations = '../03_results/out_correlations/correlation_tf_tf'
os.makedirs(dir_out_correlations, exist_ok=True)

metadata_file = os.path.join(dir_env, 'metadata_chile_cont.tsv')
md = pd.read_csv(metadata_file, sep='\t', index_col=0)
md_all = pd.read_csv(os.path.join(dir_env,'metadata_chile.tsv'), sep='\t', index_col=0)

omics_files = [f for f in os.listdir(dir_omics) if f.startswith('Matrix_TF_') and f.endswith('_all.tsv')]


# In[4]:


# latitude
bins = [-float('inf'), -40, -30, float('inf')]
labels = ['South', 'Center', 'North']
md_all['Latitude Bin'] = pd.cut(md_all['lat_cast'], bins=bins, labels=labels)


# In[5]:


subsample_dict = {
    'All Samples': md_all.index.tolist(),
    'SRF Samples': md_all[md_all['Depth level'] == 'SRF'].index.tolist(),
    'EPI Samples': md_all[md_all['Depth level'] == 'EPI'].index.tolist(),
    'MES Samples': md_all[md_all['Depth level'] == 'MES'].index.tolist(),
    'South Samples': md_all[md_all['Latitude Bin'] == 'South'].index.tolist(),
    'Center Samples': md_all[md_all['Latitude Bin'] == 'Center'].index.tolist(),
    'North Samples': md_all[md_all['Latitude Bin'] == 'North'].index.tolist(),
    'Oxic Samples': md_all[md_all['Oxygen level'] == 'OXIC'].index.tolist(),
    'Hypoxic Samples': md_all[md_all['Oxygen level'] == 'HYPOXIC'].index.tolist(),
    'Anoxic Samples': md_all[md_all['Oxygen level'] == 'ANOXIC'].index.tolist(),
    'Oxic EPI Samples': md_all[md_all['Oxy_depth'] == 'Oxic EPI'].index.tolist(),
    'Oxic MES Samples': md_all[md_all['Oxy_depth'] == 'Oxic MES'].index.tolist(),
    'SRF Oxy Samples': md_all[md_all['Oxy_depth'] == 'SRF'].index.tolist(),
    'ZMO Samples': md_all[md_all['Oxy_depth'] == 'ZMO'].index.tolist()
}

for omics_file in omics_files:
    omics_path = os.path.join(dir_omics, omics_file)
    omics_data = pd.read_csv(omics_path, sep='\t', index_col=0)
    regulatory_matrix = os.path.splitext(omics_file)[0].split('_')[-2]
    
    for subsample_name, subsample_samples in subsample_dict.items():
        aligned_omics = omics_data.loc[subsample_samples].dropna(how='all')
        if aligned_omics.empty:
            continue
        
        clr_omics = clr_(aligned_omics)
        corr_matrix = clr_omics.corr(method='spearman')

        row_linkage = linkage(pdist(corr_matrix), method='average')
        col_linkage = linkage(pdist(corr_matrix.T), method='average')
        row_dendro = dendrogram(row_linkage, no_plot=True)
        col_dendro = dendrogram(col_linkage, no_plot=True)

        reordered_corr = corr_matrix.iloc[row_dendro['leaves'], col_dendro['leaves']]

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_facecolor('white')
        
        ax.set_xticks(np.arange(len(reordered_corr.columns)))
        ax.set_yticks(np.arange(len(reordered_corr.index)))
        ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)
        ax.set_yticklabels(reordered_corr.index, fontsize=13, color="black")
        ax.set_xticklabels(reordered_corr.columns, fontsize=13, color="black", rotation=90)
        
        ax.set_xticks(np.arange(len(reordered_corr.columns) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(reordered_corr.index) + 1) - .5, minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1)
        rect = Rectangle((-.5, -.5), len(reordered_corr.columns), len(reordered_corr.index), linewidth=2, edgecolor='lightgray', facecolor='none')
        ax.add_patch(rect)

        norm = plt.Normalize(-1, 1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
        for i in range(len(reordered_corr.columns)):
            for j in range(len(reordered_corr.index)):
                correlation = reordered_corr.iat[j, i]
                color = sm.to_rgba(correlation)
                size = abs(correlation) * 1
                rect = Rectangle(xy=(i - size / 2, j - size / 2), width=size, height=size, facecolor=color)
                ax.add_patch(rect)

        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.04)
        cbar.set_label('Correlation')

        title = f'Clustered Correlation Matrix for {regulatory_matrix} {subsample_name}'
        ax.set_title(title, fontsize=16)
        
        output_file = f'correlation_clustermap_{regulatory_matrix}_{subsample_name.replace(" ", "_").lower()}'
        plt.savefig(f'{dir_out_correlations}/{output_file}.pdf', bbox_inches='tight')
        plt.close()
        

