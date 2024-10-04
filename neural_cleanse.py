import os
import time
import pickle
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torchvision
from datasets import *
from torchvision import datasets
import torchvision.transforms as transforms

from param import param


batch_size = param.train.batch_size
num_workers = param.train.num_workers

print(f"Is GPU available: {torch.cuda.is_available()}")
device = torch.device(f"cuda:{param.GPU_num}" if torch.cuda.is_available() else "cpu")

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES
#{'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9}
class_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

# Set base seed
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

# Selected dataset
# dataset = datasets.CIFAR10
dataset_name = "SpeechCommands_v1"

# Parameters
PARAMS = {
  # 'INPUT_SHAPE': (IMG_COLOR, IMG_ROWS, IMG_COLS),
  'INPUT_SHAPE': (1, 1, 80, 87),
  'LR': 0.001,
  'LAMBDA': 0.1,          # Regularization weight for the mask; higher lambda, more weight in controlling the trigger size
  'EPOCHS': 100,           # Number of epochs for optimization
  'ITER': 150,            # Number of iterations per epoch
  'BATCH_SIZE': batch_size,       # Batch size for each iteration
  'PATIENCE': 75,         # Number of batch iterations before early stopping
  'INTERVAL': 40,         # The interval at which to print the total loss during optimization
  'CMAP_P': 'viridis',    # Color map for visualizing the pattern; consider the IMG_COLOR
  'CMAP_M': 'gray',       # Color map for visualizing the mask
  'CMAP_T': 'viridis',    # Color map for visualizing the trigger; consider the IMG_COLOR
  'BG_COLOR_P': 'black',  # Background color for the pattern plot
  'BG_COLOR_M': 'black',  # Background color for the mask plot
  'BG_COLOR_T': 'black',  # Background color for the trigger plot
  'NUM_TRIALS': 3,        # Number of reverse-engineered triggers per target label
  'PATTERN_MIN': 0,       # Min pixel of initial pattern
  'PATTERN_MAX': 1,       # Max pixel of initial pattern
  'MASK_MIN': 0,          # Min pixel of initial mask
  'MASK_MAX': 1,          # Max pixel of initial mask
  # 'MAD_THRESHOLD': 2      # Threshold for the anomaly index (MAD outlier detection)
  'MAD_THRESHOLD': 0.1      # Threshold for the anomaly index (MAD outlier detection)
}

# Define the transformations for the dataset
# transform = transforms.Compose([
#   transforms.ToTensor(),                  # Convert to tensor
#   transforms.Normalize((0.5,), (0.5,))    # Normalize
# ])
transform = None
trainset = SpeechCommandsDataset(param.path.benign_train_npypath)
transform_test = None
testset = SpeechCommandsDataset(param.path.benign_test_npypath)

# Load training data
# trainset = dataset(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# Load testing data
# testset = dataset(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

_model_name = '000.pth'
benign_model_name = 'models/train/resnet34_epochs_200_batchsize_128_Adam_lr_0.02_mom_0.9_id_0.pth'
attack_model_name = './models/train/' + _model_name
save_model_name_benign = f'models/defense/NAD_benign_' + _model_name
save_model_name_poi = f'models/defense/NAD_poi_' + _model_name

def load_model(path):
    print('Loading a pretrained model')
    device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.to(device)
    return model

model = load_model(attack_model_name)
model.eval()

class ReverseEngineerTrigger:
    def __init__(self, model, target_label, input_shape, learning_rate,
                lambda_reg, initial_pattern, initial_mask):
      """
      Initialize the ReverseEngineerTrigger instance.

      Parameters:
      - model (torch.nn.Module): The pre-trained model to analyze.
      - target_label (int): The target label for misclassification.
      - input_shape (tuple): The shape of the input images.
      - learning_rate (float): Learning rate for the optimizer.
      - lambda_reg (float): Regularization weight to promote mask sparsity.
      - initial_pattern (torch.Tensor): Initial trigger pattern.
      - initial_mask (torch.Tensor): Initial mask.
      """

      self.model = model
      self.target_label = target_label
      self.learning_rate = learning_rate
      self.lambda_reg = lambda_reg
      self.pattern = torch.nn.Parameter(initial_pattern).to(device)
      self.mask = torch.nn.Parameter(initial_mask).to(device)
      self.optimizer = optim.Adam([self.pattern, self.mask], lr = self.learning_rate)

    def objective_function(self, x, y_target):
      """
      Define the objective function for optimization.

      Parameters:
      - x (torch.Tensor): Input image batch.
      - y_target (torch.Tensor): Target label for misclassification.

      Returns:
      - tuple (of torch.Tensor): The total loss, cross-entropy loss and regularization loss.
      """

      # Apply the trigger (pattern and mask) to the input images
      x_wtrigger = (1 - self.mask[None, None, :, :]) * x + self.mask[None, None, :, :] * self.pattern

      # Compute the cross-entropy loss for misclassification
      if x_wtrigger.dim() == 5:
          x_wtrigger = x_wtrigger.squeeze(0)
      logits = self.model(x_wtrigger)
      loss_ce = F.cross_entropy(logits, y_target)

      # Compute the regularization loss (based on the L1 norm) for the mask to promote sparsity
      loss_reg = torch.sum(torch.abs(self.mask))

      # Total loss is a combination of cross-entropy and regularization loss
      loss = loss_ce + self.lambda_reg * loss_reg
      return loss, loss_ce, loss_reg


    def optimize_trigger(self, trainset, target_label,
                        epochs, iterations, batch_size,
                        patience, seed, log_interval):
      """
      Optimize the trigger using mini-batch gradient descent.

      Parameters:
      - trainset (Dataset): PyTorch dataset containing training data and labels.
      - target_label (int): The target label for misclassification.
      - epochs (int): Number of epochs for optimization.
      - iterations (int): Number of iterations per epoch.
      - batch_size (int): Batch size for each iteration.
      - patience (int): Number of batch iterations to wait for improvement before stopping.
      - seed (int): Base seed for reproducibility.
      - log_interval (int): The interval at which to print the total loss.

      Returns:
      - tuple: The optimized pattern and mask with the lowest batch loss.
      """

      best_loss = float('inf')
      patience_counter = 0
      best_pattern = self.pattern.clone().detach()      # Use the current pattern as the initial best
      best_mask = self.mask.clone().detach()            # Use the current mask as the initial best

      seed_counter = 0
      for epoch in range(epochs):
        for i in range(iterations):

          # Adjust the seed for each batch iteration
          torch.manual_seed(seed + i + epoch * iterations)

          # Initialize accumulated tensors for the batch data
          accumulated_x = torch.Tensor().to(device)

          # Keep sampling batches until we get enough samples
          while len(accumulated_x) < batch_size:

            # Adjust the seed within the accumulation loop
            torch.manual_seed(seed + i + epoch * iterations + seed_counter)

            # Extract the only batch
            sampler = RandomSampler(trainset, replacement = True, num_samples = batch_size)
            single_batch_loader = DataLoader(trainset, batch_size = batch_size, sampler = sampler)
            x, y = next(iter(single_batch_loader))

            # Select only those images where the true label is not equal to the target label
            selected_indices = (y != target_label).nonzero(as_tuple = True)[0]
            selected_x = x[selected_indices].to(device)

            accumulated_x = torch.cat((accumulated_x, selected_x), 0)[:batch_size]

            seed_counter += 1

          # At this point, accumulated_x has the intended batch size
          # Get a batch of labels where every label is set to the target label
          y_target = torch.full((batch_size,), target_label, dtype = torch.long).to(device)

          self.optimizer.zero_grad()                    # Zero out any accumulated gradients
          loss, loss_ce, loss_reg = self.objective_function(accumulated_x, y_target)

          loss.backward()                               # Compute gradients
          self.optimizer.step()                         # Update pattern and mask

          # Print progress
          if i % log_interval == 0:
            print(f"Epoch {epoch}, Batch iteration {i}:\nTotal loss: {loss.item():.4f}\nCross-entropy loss: {loss_ce.item():.4f}\nRegularization loss: {loss_reg.item():.4f}\n")

          # Check if the current loss is better than the best loss
          if loss.item() < best_loss:
            best_loss = loss.item()
            best_pattern = self.pattern.clone().detach()
            best_mask = self.mask.clone().detach()
            patience_counter = 0
          else:
            patience_counter += 1

            # Early stopping if the loss does not improve for a certain number of batch iterations
            if patience_counter >= patience:
              print("Early stopping triggered.")
              self.pattern.data = best_pattern.data
              self.mask.data = best_mask.data
              return self.pattern, self.mask

      # If loop ends without early stopping, update pattern and mask with the best ones
      self.pattern.data = best_pattern.data
      self.mask.data = best_mask.data

      return self.pattern, self.mask


    def visualize_save_trigger(self, cmap_p, cmap_m, cmap_t,
                              bg_color_p, bg_color_m, bg_color_t,
                              plot_fig = True, save_fig = False, save_path = None):
      """
      Visualize and/or save the pattern, mask and trigger. Default is to visualize only.

      Parameters:
      - cmap_p (str): Color map for visualizing the pattern.
      - cmap_m (str): Color map for visualizing the mask.
      - cmap_t (str): Color map for visualizing the trigger.
      - bg_color_p (str): Background color for the pattern plot.
      - bg_color_m (str): Background color for the mask plot.
      - bg_color_t (str): Background color for the trigger plot.
      - plot_fig (bool): Whether to plot the trigger figure. Default is True.
      - save_fig (bool): Whether to save the trigger figure. Default is False.
      - save_path (str): The path where the trigger image will be saved (as a PNG file).
                        Default is None. If save_fig is True, the save_path needs to be included.
      """

      # Convert the tensors to numpy arrays for visualization
      pattern_data = self.pattern.detach().cpu().numpy()
      mask_data = self.mask.detach().cpu().numpy()

      plt.figure(figsize = (12, 5))

      # Plotting the Pattern
      ax1 = plt.subplot(1, 3, 1)
      if pattern_data.ndim == 3:
          plt.imshow(pattern_data.transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_p)  # Convert from CHW to HWC for visualization
      else:
          plt.imshow(pattern_data[0].transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_p)  # Handle case for 4D input
      plt.title("Pattern")
      plt.colorbar(fraction = 0.046, pad = 0.04, boundaries = np.linspace(0, 1, 11), ticks = np.linspace(0, 1, 11))
      plt.axis('off')
      ax1.set_facecolor(bg_color_p)

      # Plotting the Mask
      ax2 = plt.subplot(1, 3, 2)
      plt.imshow(mask_data.transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_m)
      plt.title("Mask")
      plt.colorbar(fraction = 0.046, pad = 0.04, boundaries = np.linspace(0, 1, 11), ticks = np.linspace(0, 1, 11))
      plt.axis('off')
      ax2.set_facecolor(bg_color_m)
      combined_trigger = pattern_data * mask_data[None, :, :]
      if combined_trigger.ndim == 3:
          plt.imshow(combined_trigger.transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_t)  # Convert from CHW to HWC for visualization
      else:
          plt.imshow(combined_trigger[0].transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_t)  # Handle case for 4D input
      ax3 = plt.subplot(1, 3, 3)
      combined_trigger = pattern_data * mask_data[None, :, :]
      if combined_trigger.ndim == 3:
          plt.imshow(combined_trigger.transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_t)  # Convert from CHW to HWC for visualization
      else:
          plt.imshow(combined_trigger[0].transpose(1, 2, 0), vmin = 0, vmax = 1, cmap = cmap_t)  # Handle case for 4D input
      plt.title("Trigger")
      plt.colorbar(fraction = 0.046, pad = 0.04, boundaries = np.linspace(0, 1, 11), ticks = np.linspace(0, 1, 11))
      plt.axis('off')
      ax3.set_facecolor(bg_color_t)

      plt.tight_layout()

      # Show the plot if plot_fig is True
      if plot_fig:
        plt.show()

      # Save the figure if save_fig is True and save_path is not None
      if save_fig and save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
        plt.close()


def reverse_engineer_triggers_for_all_labels(trainset, trainloader, model, params, seed):
  """
  Reverse-engineer triggers for all labels in the dataset.

  Parameters:
  - trainset (Dataset): Dataset containing training data and labels.
  - trainloader (DataLoader): DataLoader for training data.
  - model (torch.nn.Module): The pre-trained model to analyze.
  - params (dict): Dictionary of parameters.
  - seed (int): Base seed for reproducibility.

  Returns:
  - dict: Dictionary containing a list of reverse-engineered triggers for each label.
  """
  
  # Get all class labels in the dataset
  # labels = torch.unique(torch.tensor(trainloader.dataset.targets))
  labels = torch.arange(10)

  # Initialize dictionary to hold the reverse-engineered triggers for each label
  triggers = {}
  total_start_time = time.time()

  # Iterate through all the class labels
  for label in labels:

    # Initialize list for the reverse-engineered triggers
    triggers[label.item()] = []

    # Run optimization for different initial triggers and masks
    for i in range(params['NUM_TRIALS']):
      print(f"\n----- Optimizing for label {label.item()}, trial {i+1}/{params['NUM_TRIALS']} -----\n")

      # Modify the seed for each label and trial
      torch.manual_seed(seed + label.item() * params['NUM_TRIALS'] + i)

      # Create random initial trigger pattern and mask
      pattern_data = torch.rand(params['INPUT_SHAPE'], requires_grad = True).to(device)
      mask_data = torch.rand(params['INPUT_SHAPE'][1:], requires_grad = True).to(device)

      initial_pattern = pattern_data * (params['PATTERN_MAX'] - params['PATTERN_MIN']) + params['PATTERN_MIN']
      initial_mask = mask_data * (params['MASK_MAX'] - params['MASK_MIN']) + params['MASK_MIN']

      # Create trigger instance
      trigger_engineer = ReverseEngineerTrigger(
        model = model,
        target_label = label.item(),
        input_shape = params['INPUT_SHAPE'],
        learning_rate = params['LR'],
        lambda_reg = params['LAMBDA'],
        initial_pattern = initial_pattern,
        initial_mask = initial_mask
      )

      start_time = time.time()

      # Optimize the trigger
      trigger_engineer.optimize_trigger(
        trainset = trainset,
        target_label = label.item(),
        epochs = params['EPOCHS'],
        iterations = params['ITER'],
        batch_size = params['BATCH_SIZE'],
        patience = params['PATIENCE'],
        seed = seed,
        log_interval = params['INTERVAL']
      )

      end_time = time.time()
      elapsed_time = end_time - start_time
      print(f"Optimization time: {elapsed_time:.2f} seconds\n")

      # Visualise the optimized trigger
      trigger_engineer.visualize_save_trigger(
        cmap_p = params['CMAP_P'],
        cmap_m = params['CMAP_M'],
        cmap_t = params['CMAP_T'],
        bg_color_p = params['BG_COLOR_P'],
        bg_color_m = params['BG_COLOR_M'],
        bg_color_t = params['BG_COLOR_T']
      )

      # Append optimized trigger to the list
      triggers[label.item()].append(trigger_engineer)

  total_end_time = time.time()
  total_elapsed_time = total_end_time - total_start_time
  minutes = int(total_elapsed_time // 60)
  seconds = int(total_elapsed_time % 60)
  print(f"\n\nTotal optimization time: {minutes} minutes, {seconds} seconds.")

  # Reset seed
  torch.manual_seed(seed)

  return triggers

reverse_engineered_triggers = reverse_engineer_triggers_for_all_labels(
  trainset = trainset,
  trainloader = trainloader,
  model = model,
  params = PARAMS,
  seed = SEED
)
# print(reverse_engineered_triggers)

def calculate_attack_success_rate(model, testloader, trigger_pattern, trigger_mask, target_label):
  """
  Calculate the Attack Success Rate (ASR) for a given trigger using batches of test data from a DataLoader.

  Parameters:
  - model (torch.nn.Module): The model under attack.
  - testloader (DataLoader): DataLoader for the test data.
  - trigger_pattern (torch.Tensor): The trigger pattern to apply.
  - trigger_mask (torch.Tensor): The trigger mask to apply.
  - target_label (int): The label to which inputs should be misclassified.

  Returns:
  - float: The ASR of the trigger.
  """
  
  total_attacks = 0
  successful_attacks = 0

  for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)

    # Filter images and labels where true label is not equal to target label
    valid_indices = (labels != target_label).nonzero(as_tuple = True)[0]
    images = images[valid_indices]
    labels = labels[valid_indices]

    # Apply the trigger to the images
    images_triggered = (1 - trigger_mask[None, :, :]) * images + trigger_mask[None, :, :] * trigger_pattern
    if images_triggered.shape[0] == 0:
      continue
    # print(f'images_triggered_shape: {images_triggered.shape}')

    # Get the model's predictions
    logits = model(images_triggered)
    predictions = torch.argmax(logits, dim = 1)

    # Update the counts for successful attacks and total attacks
    successful_attacks += torch.sum(predictions == target_label).item()
    total_attacks += len(images)

  # Calculate the ASR
  asr = successful_attacks / total_attacks if total_attacks > 0 else 0

  return asr


def select_best_trigger(model, testloader, reverse_engineered_triggers):
  """
  Select the best reverse-engineered trigger for each label based on the highest ASR using a DataLoader.

  Parameters:
  - model (torch.nn.Module): The model under attack.
  - testloader (DataLoader): DataLoader for the test data.
  - reverse_engineered_triggers (dict): Dictionary containing reverse-engineered triggers for each label.

  Returns:
  - dict: Dictionary in which the key is the target label
          and the value is another dictionary with
          the best trigger object, its ASR, and mask L1 norm.
  """

  best_triggers = {}

  # Iterate through each target label and its corresponding triggers
  for target_label, triggers_list in reverse_engineered_triggers.items():
    max_asr = 0
    best_trigger = None
    min_l1_norm = float('inf')

    # Evaluate each trigger's ASR
    for trigger_obj in triggers_list:
      trigger_pattern = trigger_obj.pattern
      trigger_mask = trigger_obj.mask

      asr = calculate_attack_success_rate(
        model = model,
        testloader = testloader,
        trigger_pattern = trigger_pattern,
        trigger_mask = trigger_mask,
        target_label = target_label
      )

      # Calculate the L1 norm of the current trigger mask
      current_l1_norm = torch.sum(torch.abs(trigger_mask)).item()

      # Update the best trigger if the current trigger has a higher ASR or
      # if the ASR is the same but the mask L1 norm is smaller (and not zero)
      if asr > max_asr or (asr == max_asr and 0 < current_l1_norm < min_l1_norm):
        max_asr = asr
        best_trigger = trigger_obj
        min_l1_norm = current_l1_norm

      # Store the best trigger for the current label
      best_triggers[target_label] = {'Trigger': best_trigger, 'ASR': max_asr, 'Mask L1 norm': min_l1_norm}

  return best_triggers

best_triggers = select_best_trigger(
  model = model,
  testloader = testloader,
  reverse_engineered_triggers = reverse_engineered_triggers
)

# Visualize the best triggers
print("\n----- Best triggers for each class -----\n\n")

for target_label, best_trigger_dict in best_triggers.items():
  print(f"Label: {idx_to_class[target_label]}, Attack Success Rate: {best_trigger_dict['ASR'] * 100:.2f}%\n")

  best_trigger_dict['Trigger'].visualize_save_trigger(
    cmap_p = PARAMS['CMAP_P'],
    cmap_m = PARAMS['CMAP_M'],
    cmap_t = PARAMS['CMAP_T'],
    bg_color_p = PARAMS['BG_COLOR_P'],
    bg_color_m = PARAMS['BG_COLOR_M'],
    bg_color_t = PARAMS['BG_COLOR_T']
  )

  print(f"\n")
  
def mad_outlier_detection(best_triggers, threshold):
  """
  Detect outliers using the Median Absolute Deviation (MAD) method.

  Parameters:
  - best_triggers (dict): Dictionary in which the key is the target label
                          and the value is another dictionary with
                          the best trigger object and its ASR and mask L1 norm.
  - threshold (float): Threshold for the anomaly index.

  Returns:
  - outliers (dict): Dictionary of trigger
                      in which the key is the target label
                      and the value is another dictionary
                      with the best trigger object,
                      its ASR, mask L1 norm and anomaly index
                      and whether it's an outlier.
  """

  # Extract mask L1 norms from the best_triggers dictionary and convert to tensor
  l1_norms = torch.tensor([trigger_info['Mask L1 norm'] for trigger_info in best_triggers.values()])

  # Calculate the median of the mask L1 norms
  median_l1 = torch.median(l1_norms)

  # Compute the absolute deviation from the median for each mask L1 norm
  abs_deviation = torch.abs(l1_norms - median_l1)

  # Calculate the MAD
  mad = 1.4826 * torch.median(abs_deviation)

  # Calculate the anomaly index
  # Assuming the distribution is normal, a constant estimator (1.4826) is applied to normalize the anomaly index
  # Add a very small value to avoid division by zero
  anomaly_index = abs_deviation / (mad + 1e-10)

  print(f"Median of the mask L1 norms: {median_l1.item():.2f}")
  print(f"Median Absolute Deviation (MAD): {mad.item():.2f}")

  # Identify outliers
  outliers = {}
  for i, (target_label, trigger_info) in enumerate(best_triggers.items()):
    if anomaly_index[i].item() > threshold:
      outliers[target_label] = {
        'Trigger': trigger_info['Trigger'],
        'ASR': trigger_info['ASR'],
        'Mask L1 norm': trigger_info['Mask L1 norm'],
        'Anomaly index': anomaly_index[i].item(),
        'Outlier': 'Yes'
      }
    else:
      outliers[target_label] = {
        'Trigger': trigger_info['Trigger'],
        'ASR': trigger_info['ASR'],
        'Mask L1 norm': trigger_info['Mask L1 norm'],
        'Anomaly index': anomaly_index[i].item(),
        'Outlier': 'No'
      }

  return outliers

# Calculate anomaly index for all best triggers
best_trigger_outliers = mad_outlier_detection(
  best_triggers = best_triggers,
  threshold = PARAMS['MAD_THRESHOLD']
)

# Filter for outliers
final_triggers = {label: info for label, info in best_trigger_outliers.items() if info['Outlier'] == 'Yes'}

# Visualize the final triggers outliers
print("\n----- Final triggers -----")

for target_label, trigger_info in final_triggers.items():
  print(f"\nLabel {idx_to_class[target_label]}:")
  print(f"- ASR: {trigger_info['ASR'] * 100:.2f}%")
  print(f"- Mask L1 norm: {trigger_info['Mask L1 norm']:.2f}")
  print(f"- Anomaly index: {trigger_info['Anomaly index']:.2f}")
  print(f"- Outlier: {trigger_info['Outlier']}\n")

  trigger_info['Trigger'].visualize_save_trigger(
    cmap_p = PARAMS['CMAP_P'],
    cmap_m = PARAMS['CMAP_M'],
    cmap_t = PARAMS['CMAP_T'],
    bg_color_p = PARAMS['BG_COLOR_P'],
    bg_color_m = PARAMS['BG_COLOR_M'],
    bg_color_t = PARAMS['BG_COLOR_T']
  )

best_trigger_outliers = mad_outlier_detection(
  best_triggers = best_triggers,
  threshold = PARAMS['MAD_THRESHOLD']
)

# Filter for outliers
final_triggers = {label: info for label, info in best_trigger_outliers.items() if info['Outlier'] == 'Yes'}

# Visualize the final triggers outliers
print("\n----- Final triggers -----")

for target_label, trigger_info in final_triggers.items():
  # print(type(target_label))
  print(f"\nLabel {idx_to_class[int(target_label)]}:")
  print(f"- ASR: {trigger_info['ASR'] * 100:.2f}%")
  print(f"- Mask L1 norm: {trigger_info['Mask L1 norm']:.2f}")
  print(f"- Anomaly index: {trigger_info['Anomaly index']:.2f}")
  print(f"- Outlier: {trigger_info['Outlier']}\n")

  trigger_info['Trigger'].visualize_save_trigger(
    cmap_p = PARAMS['CMAP_P'],
    cmap_m = PARAMS['CMAP_M'],
    cmap_t = PARAMS['CMAP_T'],
    bg_color_p = PARAMS['BG_COLOR_P'],
    bg_color_m = PARAMS['BG_COLOR_M'],
    bg_color_t = PARAMS['BG_COLOR_T']
  )
  
def log_experiment_results(model_name, dataset_name, seed, params, final_triggers, log_directory):
  """
  Log the experiment results including model info, parameters and trigger visualizations.

  Parameters:
  - model_name (str): Filename of the model being evaluated.
  - dataset_name (str): Name of the dataset used.
  - seed (int): Seed value used.
  - params (dict): Dictionary of parameters used.
  - final_triggers (dict): Dictionary of final triggers.
  - log_directory (str): Directory where the logs will be saved.
  """

  # Define the logging directory
  current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  current_log_dir = os.path.join(log_directory, f'nc_{current_time}')

  # Create a new directory for the current run
  if not os.path.exists(current_log_dir):
    os.makedirs(current_log_dir)

  # Define the path for the .txt file
  txt_file_path = os.path.join(current_log_dir, 'info.txt')

  # Write information to the .txt file
  with open(txt_file_path, 'w') as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Seed: {seed}\n")

    f.write("\n----- Parameters -----\n\n")
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

    f.write("\n----- Triggers Info -----\n")
    for target_label, trigger_info in final_triggers.items():
      f.write(f"\nLabel {target_label}:\n")
      f.write(f"- ASR: {trigger_info['ASR'] * 100:.2f}%\n")
      f.write(f"- Mask L1 norm: {trigger_info['Mask L1 norm']:.2f}\n")
      f.write(f"- Anomaly index: {trigger_info['Anomaly index']:.2f}\n")
      f.write(f"- Outlier: {trigger_info['Outlier']}\n")

  # Save trigger visualizations as .png files using the provided method
  for target_label, trigger_info in final_triggers.items():
    trigger_png_path = os.path.join(current_log_dir, f"trigger_label_{target_label}.png")

    trigger_info['Trigger'].visualize_save_trigger(
      cmap_p = PARAMS['CMAP_P'],
      cmap_m = PARAMS['CMAP_M'],
      cmap_t = PARAMS['CMAP_T'],
      bg_color_p = PARAMS['BG_COLOR_P'],
      bg_color_m = PARAMS['BG_COLOR_M'],
      bg_color_t = PARAMS['BG_COLOR_T'],
      plot_fig = False,
      save_fig = True,
      save_path = trigger_png_path
    )
    
# Define the LOG_DIR
log_dir = 'experiments'
model_name = attack_model_name
# Call the function to log the results
log_experiment_results(
  model_name = model_name,
  dataset_name = dataset_name,
  seed = SEED,
  params = PARAMS,
  final_triggers = final_triggers,
  log_directory = log_dir
)