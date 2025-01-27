import numpy as np
import random
import tensorflow as tf
import keras
import torch
import gzip
from google.colab import files

def set_random_seed(seed, deterministic=False):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  keras.utils.set_random_seed(seed)
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  print(f"Random seed set as {seed}")

# Display the first n entries of a dictionary
def display_dict_entries(dictionary, n=5):
    for key, value in list(dictionary.items())[:n]:
        print(f"{key}: {value}")
