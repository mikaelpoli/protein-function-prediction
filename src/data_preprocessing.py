import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Datasets
"""

def load_original_data(config, test=False):
  # Set path
  embeddings_path = config['paths']['test_embeddings'] if test else config['paths']['train_embeddings']

  # Load T5 protein embeddings from a .h5 file
  with h5py.File(embeddings_path, 'r') as f:

    # Extrect protein IDs
    ids = np.array(list(f.keys()))
    embeddings = np.array([f[protein_id][:] for protein_id in ids])

  if not test:
    # Load train set from a .tsv file
    train_set = pd.read_csv(config['paths']['train_set'], sep='\t')
    train_set.rename(columns={'Protein_ID': 'protein_id', 'GO_term': 'go_term'}, inplace=True)

    # Sort train set alphabetically (like protein embeddings)
    train_set.sort_values(by=['protein_id'], inplace=True)

    return train_set, embeddings, ids

  return embeddings, ids

def load_custom_data(config, test=False):
  # Set paths
  embeddings_path = config['paths']['custom_test_embeddings'] if test else config['paths']['custom_train_embeddings']
  embeddings_key = 'custom_test_embeddings' if test else 'custom_train_embeddings'
  dataset_path = config['paths']['custom_test_set'] if test else config['paths']['custom_train_set']
  ids_path = config['paths']['custom_test_ids'] if test else config['paths']['custom_train_ids']

  # Load T5 protein embeddings from a .h5 file
  with h5py.File(embeddings_path, 'r') as f:
    embeddings = np.array(f[embeddings_key])

  # Load dataset and IDs from a .tsv file
  dataset = pd.read_csv(dataset_path, sep='\t')
  ids_df = pd.read_csv(ids_path, sep='\t', header=None)
  ids = ids_df[0].to_numpy()

  return dataset, embeddings, ids

"""
Preprocessing
"""

def get_protein_ids(df):
  return df['protein_id'].unique()

def split_by_aspect(df):
  df_mf = df[df['aspect'] == 'molecular_function']
  df_bp = df[df['aspect'] == 'biological_process']
  df_cc = df[df['aspect'] == 'cellular_component']
  return df_mf, df_bp, df_cc

# Prepare training set labels
def get_labels_dict(df):
  labels_dict = df.groupby('protein_id')['go_term'].apply(list).to_dict()
  return labels_dict


# Binarize training labels for multi-label classifiction
def binarize_labels(labels_dict):
  labels_list = list(labels_dict.values())
  mlb = MultiLabelBinarizer()
  labels_bin = mlb.fit_transform(labels_list)
  return labels_bin, mlb.classes_

"""
Data prparation for DNN training
"""
def build_custom_datasets(train_ids, train_embeddings, custom_test_ids):
  # Get the custom test set protein embeddings and indices
  sampled_indices = np.isin(train_ids, custom_test_ids)
  custom_test_idx = np.where(sampled_indices)[0]
  custom_test_embeddings = train_embeddings[custom_test_idx]

  # Get the updated training set protein embeddings, indices, and protein IDs
  custom_train_idx = np.where(~sampled_indices)[0]
  custom_train_embeddings = train_embeddings[custom_train_idx]
  custom_train_ids = train_ids[~np.isin(train_ids, custom_test_ids)]

  return custom_test_idx, custom_test_embeddings, custom_train_ids, custom_train_idx, custom_train_embeddings

# Prepare input features and labels for each custom aspect training set
def prepare_features_and_labels(aspect, custom_aspect_dataset, custom_train_embeddings, custom_full_train_ids):
  print(f"Preparing {aspect} features and labels")

  # Locate indices in training set
  id_set = set(custom_aspect_dataset['protein_id'].unique())
  idx = np.array([id_ in id_set for id_ in custom_full_train_ids])

  # Get indices from training set
  custom_train_aspect_idx = np.where(idx)[0]
  custom_train_aspect_ids = custom_full_train_ids[custom_train_aspect_idx]

  # Get protein embeddings corresponding to indices and save ID-embedding correspondence
  x = custom_train_embeddings[custom_train_aspect_idx]
  custom_train_aspect_dict = dict(zip(custom_train_aspect_ids, x))

  # Prepare labels
  y = get_labels_dict(custom_aspect_dataset)
  y_bin, classes = binarize_labels(y)

  return x, y_bin, classes

def plot_label_distribution(train_labels, val_labels, aspect, bin_size=30):
  # Count label frequencies
  train_label_distribution = np.sum(train_labels, axis=0)
  test_label_distribution = np.sum(val_labels, axis=0)

  # Calculate the number of bins to ensure divisibility
  num_bins = train_label_distribution.shape[0] // bin_size

  # Reshape the distributions based on the calculated number of bins
  train_bins = np.sum(train_label_distribution[:num_bins * bin_size].reshape(-1, bin_size), axis=1)
  val_bins = np.sum(test_label_distribution[:num_bins * bin_size].reshape(-1, bin_size), axis=1)
  bin_indices = np.arange(len(train_bins))

  plt.bar(bin_indices - 0.2, train_bins, width=0.4, label='Train', color='blue', alpha=0.7)
  plt.bar(bin_indices + 0.2, val_bins, width=0.4, label='Val', color='orange', alpha=0.7)
  plt.title(f"Binned Label Distribution for {aspect}")
  plt.xlabel('Label Bins')
  plt.ylabel('Frequency')
  plt.legend()
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.show()

  return plt

"""
Data preparation for BLAST-based scores
"""
# Process GO terms from the training dataset.
def process_go_terms(train_set):
  go_terms = [[term] for term in train_set['go_term'].unique()]
  sorted_go_terms = sorted(go_terms)

  mlb = MultiLabelBinarizer()
  binary_go_matrix = mlb.fit_transform(sorted_go_terms)

  # Create a dictionary with GO term labels as keys and binary vectors as values
  go_terms_dict = {
      go_term[0]: np.array(binary_vector.tolist())
      for go_term, binary_vector in zip(sorted_go_terms, binary_go_matrix)
  }

  return go_terms_dict

# Process BLAST results by filtering, weighting, and normalizing
def process_blast_results(train_blast, custom_test_ids, evalue_threshold=1e-5, top_n=50, max_value=10):
  """
  Args:
      train_blast (pd.DataFrame): A DataFrame containing BLAST results with columns 'query', 'target', and 'evalue'.
      custom_test_ids (set): A set of protein IDs in the custom test set.
      evalue_threshold (float, optional): The maximum e-value threshold for filtering. Default is 1e-5.
      top_n (int, optional): The maximum number of matches to keep for each query protein. Default is 50.
      max_value (float, optional): The maximum weight value. Default is 10.

  Returns:
      pd.DataFrame: A processed DataFrame with filtered, weighted, and normalized BLAST results.
  """
  # Keep BLAST results only for the proteins in the custom test set
  blast = train_blast[train_blast["query"].isin(custom_test_ids)]

  # Remove BLAST results where the target protein is in the custom test set
  blast = blast[~blast["target"].isin(custom_test_ids)]

  # Filter BLAST results based on e-value and keep only the top N matches
  blast = blast[blast["evalue"] <= evalue_threshold]
  blast = (
      blast.sort_values(by=["query", "evalue"])
      .groupby("query")
      .head(top_n)
  )

  # Calculate weights based on e-value
  blast.loc[:, "weight"] = np.minimum(
      -np.log10(blast.loc[:, "evalue"] + 1e-300), 1e6
  )

  # Normalize weights for each query protein
  blast["weight"] = blast.groupby("query")["weight"].transform(
      lambda x: x / x.sum() * max_value
  )

  return blast

# Create a dictionary mapping BLAST target protein IDs to their corresponding GO terms
def create_target_go_terms_dict(target_go_terms):
  """
  Args:
      target_go_terms (pd.DataFrame): A DataFrame where the index contains protein IDs
                                      and the rows contain GO terms.
  """
  return {
      protein_id: target_go_terms.iloc[i]
      for i, protein_id in enumerate(target_go_terms.index)
  }

"""
Process predictions for .tsv export
"""

def process_predictions(combined_predictions, aspect, threshold=0.2, top_n=500):
  """
  Args:
      combined_predictions (pd.DataFrame): Input dataframe.
      threshold (float): Minimum prediction value to retain rows.
      top_n (int): Number of top predictions to retain per group.

  Returns:
      pd.DataFrame: Processed dataframe.
  """
  df = combined_predictions[combined_predictions['aspect']==aspect]
  df_sorted = df.sort_values(by=['protein_id', 'prediction'], ascending=[True, False])
  df_limited = df_sorted.groupby('protein_id').head(top_n)
  df_limited.loc[:, 'prediction'] = df_limited['prediction'].round(3)

  return df_limited[df_limited['prediction'] > threshold]
