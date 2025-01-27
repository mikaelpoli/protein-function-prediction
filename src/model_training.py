import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from functools import partial
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout, ReLU, LeakyReLU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

def build_model(input_dim, output_dim, aspect, model_type, n_hidden, size_hidden, activation_hidden, dropout, leaky_relu_alpha, set_seed=42):
  """
  Args:
    model_type (str) can be 'decreasing' or 'fixed'. "decreasing" will create an architecture where evey hidden layer is half the size of its predecessor. "fixed"
    will create an architecture where all hidden layers have the same size.
    aspect (str) can be 'molecular_function', 'biological_process', or 'cellular_component'
    activation_hidden (str) can be 'relu' or 'leaky_relu'
  """
  # Ensure model type is specified
  if model_type not in ('decreasing', 'fixed'):
      raise ValueError("Model type must be 'decreasing' or 'fixed'")
  # Ensure aspect is specified
  if aspect not in ('molecular_function', 'biological_process', 'cellular_component'):
      raise ValueError("Aspect must be 'molecular_function' or 'biological_process' or 'cellular_component'")
  # Ensure activation function is specified
  if activation_hidden not in ('relu', 'leaky_relu'):
      raise ValueError("Activation for hidden layers must be 'relu' or 'leaky_relu'")

  # Input, dropout, and batch normalization layers
  inputs = Input(shape=(input_dim,), name='input_layer')
  x = BatchNormalization(name='layer_0_batch_norm')(inputs)
  x = Dropout(dropout, name='layer_0_dropout')(x)

  # Hidden layers for decreasing model
  if model_type == 'decreasing':
    dense_layer = partial(Dense, kernel_initializer=HeNormal(seed=set_seed), use_bias=True)
    for i in range(n_hidden):
        x = dense_layer(size_first_hidden, name=f'layer_{i+1}_dense')(x)
        if activation_hidden == 'relu':
          x = ReLU(name=f'layer_{i+1}_relu')(x)
        else:
          x = LeakyReLU(negative_slope=leaky_relu_alpha, name=f'layer_{i+1}_leaky_relu')(x)
        x = BatchNormalization(name=f'layer_{i+1}_batch_norm')(x)
        x = Dropout(dropout, name=f'layer_{i+1}_dropout')(x)
        size_first_hidden = size_first_hidden // 2

  # Hidden layers for fixed model
  elif model_type == 'fixed':
    dense_layer = partial(Dense, kernel_initializer=HeNormal(seed=set_seed), use_bias=True)
    for i in range(n_hidden):
        x = dense_layer(size_hidden, name=f'layer_{i+1}_dense')(x)
        if activation_hidden == 'relu':
          x = ReLU(name=f'layer_{i+1}_relu')(x)
        else:
          x = LeakyReLU(negative_slope=leaky_relu_alpha, name=f'layer_{i+1}_leaky_relu')(x)
        x = BatchNormalization(name=f'layer_{i+1}_batch_norm')(x)
        x = Dropout(dropout, name=f'layer_{i+1}_dropout')(x)

  # Output layer
  output = Dense(output_dim, activation='sigmoid', name='output_layer_sigmoid')(x)

  # Create the model
  model = Model(inputs=inputs, outputs=output, name=f'{aspect}_{model_type}')
  return model

def plot_metrics(history):
    # Plot loss
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)  # 1 row, 3 columns; first plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot AUC-ROC
    plt.subplot(1, 3, 2)  # 1 row, 3 columns; second plot
    plt.plot(history.history['auc-roc'], label='Train AUC-ROC')
    plt.plot(history.history['val_auc-roc'], label='Validation AUC-ROC')
    plt.title('AUC ROC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.legend()

    # Plot AUC-PR
    plt.subplot(1, 3, 3)  # 1 row, 3 columns; third plot
    plt.plot(history.history['auc-pr'], label='Train AUC-PR')
    plt.plot(history.history['val_auc-pr'], label='Validation AUC-PR')
    plt.title('AUC Precision-Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-PR')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(X_train, y_train, X_val, y_val, aspect, model_type, n_hidden, size_hidden, activation, dropout, learning_rate, epochs, batch_size, leaky_relu_alpha=0.01, set_seed=42):
  # Ensure type of model is provided
  if model_type not in ('decreasing', 'fixed'):
      raise ValueError("Type must be 'decreasing' or 'fixed'")
  # Ensure activation function is provided
  if activation not in ('relu', 'leaky_relu'):
      raise ValueError("Type must be 'relu' or 'leaky_relu'")
  keras.utils.set_random_seed(set_seed)
  input_dim = X_train.shape[1]
  output_dim = y_train.shape[1]

  # Build model
  model = build_model(input_dim, output_dim, aspect, model_type, n_hidden, size_hidden, activation, dropout, leaky_relu_alpha, set_seed)

  # Print model configuration
  print("=" * 40)
  print(f"{'Model Configuration':^40}")
  print("=" * 40)
  print(f"Aspect:               {aspect}")
  print(f"Model Type:           {model_type}")
  print(f"Hidden Layers:        {n_hidden}")
  if model_type == 'decreasing':
      print(f"First Layer Units:    {size_hidden}")
  elif model_type == 'fixed':
      print(f"Units Per Layer:      {size_hidden}")
  print(f"Dropout:              {dropout}")
  print(f"Activation:           {activation}")
  if activation == 'leaky_relu':
      print(f"Leaky ReLU Alpha:     {leaky_relu_alpha}")
  print(f"Learning Rate:        {learning_rate}")
  print("=" * 40)

  # Implement early stopping and adaptive learning rate
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

  model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(learning_rate=learning_rate),
  metrics=[
      tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
      tf.keras.metrics.Precision(name="precision"),
      tf.keras.metrics.Recall(name="recall"),
      tf.keras.metrics.AUC(curve='ROC', name="auc-roc", multi_label=True),
      tf.keras.metrics.AUC(curve='PR', name="auc-pr", multi_label=True)
      ]
  )

  # Record performance
  model.history = model.fit(
      X_train,
      y_train,
      validation_data=(X_val, y_val),
      epochs=epochs,
      batch_size=batch_size,
      callbacks=[early_stopping, reduce_lr],
      verbose=1
  )

  # Plot training and validation metrics
  plot_metrics(model.history)
  return model

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

def build_predictions_df(predictions_array, protein_ids, classes, aspect):
  protein_id_list = []
  aspect_list = []
  go_term_list = []
  prediction_list = []

  # Loop through each protein and its predictions
  for i, protein_id in enumerate(protein_ids):
      # For each protein, loop through the GO terms
      for j, go_term in enumerate(classes):
          protein_id_list.append(protein_id)
          aspect_list.append(aspect)  # All rows will have the same aspect
          go_term_list.append(go_term)
          prediction_list.append(predictions_array[i, j])

  # Create the DataFrame
  df = pd.DataFrame({
      'protein_id': protein_id_list,
      'aspect': aspect_list,
      'go_term': go_term_list,
      'prediction': prediction_list
  })
  return df
