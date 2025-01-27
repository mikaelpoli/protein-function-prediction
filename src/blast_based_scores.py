import pandas as pd
import numpy as np
from scipy.special import expit

# Calculate BLAST-based scores for a given set of test queries
def calculate_blast_based_scores(test_ids_list, blast_results_df, go_encodings_dict, target_go_terms_dict, n_go_terms, scores_init=-3.0):
  """
  This function computes scores for each query in the "test_ids_list" by aggregating weighted contributions from BLAST matches.
  The scores are then transformed into probabilities using the sigmoid function.

  Args:
      test_ids_list (list): A list of test query identifiers for which scores need to be calculated.
      blast_results_df (pandas.DataFrame): A DataFrame containing BLAST results with the following columns:
          - 'query': Query identifier.
          - 'target': Target identifier (matched sequence).
          - 'weight': Weight assigned to the match (e.g., alignment score or similarity score).

      go_encodings_dict (dict): A dictionary where keys are GO terms, and values are their corresponding numeric encodings (e.g., vectors or weights).
      target_go_terms_dict (dict): A dictionary mapping target identifiers to a list of associated GO terms.
      n_go_terms (int): The total number of GO terms considered. Determines the size of the score vector for each query.
      scores_init (float, optional (default=-3.0)): The initial value for the score vector. Typically a negative value to represent low initial probabilities.

  Returns:
      dict: A dictionary where keys are query identifiers from `test_ids_list`, and values are 1D numpy arrays containing
      the calculated scores (as probabilities) for each GO term.

  Notes:
  - If a query from "test_ids_list" does not have any matches in `blast_results_df`, it will not appear in the returned dictionary.
  - The sigmoid transformation ensures scores are in the range [0, 1], making them interpretable as probabilities.
  """
  blast_based_scores = {}
  queries = blast_results_df['query'].unique()

  for query in test_ids_list:
    if query in queries:
      matches = blast_results_df[blast_results_df['query'] == query]
      scores = np.full(n_go_terms, scores_init)
      for _, row in matches.iterrows():
          target_id = row["target"]
          weight = row["weight"]
          if target_id in target_go_terms_dict:
              target_go_terms = target_go_terms_dict[target_id]
              for go_term in target_go_terms:
                scores += weight * go_encodings_dict[go_term]

      # Apply sigmoid function to convert scores to probabilities
      scores = expit(scores)
      scores = np.round(scores, 3)
      blast_based_scores[query] = scores

  return blast_based_scores
