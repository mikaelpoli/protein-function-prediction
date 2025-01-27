import pandas as pd
import numpy as np

def combine_predictions(mlp_preds, blast_scores, alpha=0.5):
  # Alpha controls the contribution of MLP predictions
  return alpha * mlp_preds + (1 - alpha) * blast_scores

# Update predictions by combining MLP predictions with BLAST scores where available
def update_predictions(pred_all_sorted, blast_based_scores, alpha=0.5):
  """
  Args:
  - pred_all_sorted: DataFrame of predictions sorted by protein_id.
  - blast_based_scores: Dictionary of BLAST scores with protein_id as keys.
  - alpha: Weight for MLP predictions (default 0.5).

  Returns:
  - Updated DataFrame with combined predictions.
  """
  def process_group(group):
      protein_id = group.name
      if protein_id in blast_based_scores:
          blast_scores = blast_based_scores[protein_id]
          combined_probs = combine_predictions(group['prediction'].values, blast_scores, alpha)
      else:
          combined_probs = group['prediction'].values
      return group.assign(prediction=combined_probs)

  # Apply the process_group function to each protein_id group
  return pred_all_sorted.groupby('protein_id', group_keys=False).apply(process_group)
