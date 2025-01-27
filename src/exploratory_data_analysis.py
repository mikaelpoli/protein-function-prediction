import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns
import plotly.express as px

"""
Intersection of proteins in CC, MF, BP datasets
"""
# Get number of proteins in each of the three datasets and their intersections
def get_proteins(cc_ids, mf_ids, bp_ids):
  # Proteins for single aspects
  only_cc = set(cc_ids) - set(bp_ids) - set(mf_ids)
  only_bp = set(bp_ids) - set(cc_ids) - set(mf_ids)
  only_mf = set(mf_ids) - set(cc_ids) - set(bp_ids)

  # Proteins for intersections of two aspects
  cc_mf = (set(cc_ids) & set(mf_ids)) - set(bp_ids)
  cc_bp = (set(cc_ids) & set(bp_ids)) - set(mf_ids)
  bp_mf = (set(bp_ids) & set(mf_ids)) - set(cc_ids)

  # Proteins for intersection of all three aspects
  cc_mf_bp = set(cc_ids) & set(mf_ids) & set(bp_ids)

  return only_cc, only_bp, only_mf, cc_mf, cc_bp, bp_mf, cc_mf_bp

# Create Venn diagram
def create_venn_diagram(cc_ids, mf_ids, bp_ids, custom=False, test_set=False):
  test = 'Test' if test_set else 'Train'
  dataset_type = f'(Custom {test} Set)' if custom else f'(Original {test} Set)'

  only_cc, only_bp, only_mf, cc_mf, cc_bp, bp_mf, cc_mf_bp = get_proteins(cc_ids, mf_ids, bp_ids)
  n_cc = len(cc_ids)
  n_mf = len(mf_ids)
  n_bp = len(bp_ids)
  n_proteins = len(only_cc) + len(only_bp) + len(only_mf) + len(cc_mf) + len(cc_bp) + len(bp_mf) + len(cc_mf_bp)
  subsets = [len(only_cc), len(only_bp), len(cc_bp),
             len(only_mf), len(cc_mf), len(bp_mf),
             len(cc_mf_bp)]
  n_proteins = sum(subsets)
  percentages = [f"{(count / n_proteins) * 100:.2f}%" for count in subsets]

  colors = ['#0173b2', '#029e73', '#d55e00']
  venn = venn3(subsets=subsets,
                set_labels=(f"Cellular Component (n = {n_cc})",
                            f"Biological Process (n = {n_bp})",
                            f"Molecular Function (n = {n_mf})"),
                set_colors=colors)

  # Add percentages to the subset labels
  subset_labels = ['100', '010', '110', '001', '101', '011', '111']
  for subset, percentage in zip(subset_labels, percentages):
      if venn.get_label_by_id(subset):
          venn.get_label_by_id(subset).set_text(f"{venn.get_label_by_id(subset).get_text()}\n({percentage})")

  plt.title(f"Counts of Proteins by Aspect for {dataset_type} (Total Unique Protein IDs = {n_proteins})", fontsize=14)

  plt.show()
  return plt

def get_go_term_value_counts(df, descending=True, by_aspect=True):
  if not by_aspect:
    n = len(df['protein_id'].unique())
    go_counts = df['go_term'].value_counts()
    go_counts_percentage = round((go_counts / n * 100), 2)
    go_counts = pd.DataFrame({
        'go_term': go_counts.index,
        'count': go_counts,
        'percentage': go_counts_percentage})

    return n, go_counts

  else:
    n = df['go_term'].nunique()
    go_counts = df.groupby('aspect')['go_term'].nunique().reset_index(name='count')
    go_counts['percentage'] = round((go_counts['count'] / n * 100), 2)

    if descending:
      go_counts = go_counts.sort_values('percentage', ascending=False)

  return n, go_counts

def plot_go_by_aspect(train_set_df, custom=False, test_set=False):
  test = 'Test' if test_set else 'Train'
  dataset_type = f'(Custom {test} Set)' if custom else f'(Original {test} Set)'

  n, go_counts = get_go_term_value_counts(train_set_df, descending=True, by_aspect=True)
  fig = px.bar(go_counts,
             x='aspect',
             y='percentage',
             color='percentage',
             labels={'percentage': 'Percentage (%)', 'aspect': 'Aspect', 'count': 'Count'},
             title=f"Percentage of GO Terms by Aspect for {dataset_type} (Total Unique GO terms = {n})",
             hover_data={'count': True})
  fig.show()
  return fig

def plot_n_go(train_df, n, aspect='set_aspect', custom=False, test_set=False):
  test = 'Test' if test_set else 'Train'
  dataset_type = f'(Custom {test} Set)' if custom else f'(Original {test} Set)'

  n_rows, counts_df = get_go_term_value_counts(train_df, descending=False, by_aspect=False)
  counts_df = counts_df.head(n)
  fig = px.bar(counts_df,
            x='go_term',
            y='count',
            color='percentage',
            labels={'go_term': 'GO Term', 'count': 'Count', 'percentage': 'Percentage'},
            title=f"Counts and Frequency of {n} Most Frequent GO Terms for {aspect} {dataset_type}",
            hover_data={'percentage': True})
  fig.show()
  return fig
