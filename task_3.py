import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[0:,0] += f1
X_full[0:,1] += f2 
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3 # Or 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
phoneme_1_indices = np.where(phoneme_id==1)[0]
phoneme_2_indices = np.where(phoneme_id==2)[0]
X_phonemes_1_2 =X_full[np.append(phoneme_1_indices,phoneme_2_indices)]
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
model_npy_file = 'data/GMM_params_phoneme_0{ph}_k_0{k}.npy'
X =X_phonemes_1_2.copy()
# first model's predections
model_ph1 =np.ndarray.tolist(np.load(model_npy_file.format(ph=1, k=k), allow_pickle=True))
Z_model_ph1 = np.sum ( get_predictions(model_ph1['mu'], model_ph1['s'], model_ph1['p'], X),axis=1)

# second model's predections
model_ph2 =np.ndarray.tolist(np.load(model_npy_file.format(ph=2, k=k), allow_pickle=True))
Z_model_ph2 = np.sum(get_predictions(model_ph2['mu'], model_ph2['s'], model_ph2['p'], X),axis=1)

preds = Z_model_ph1>Z_model_ph2
y_truth = np.append(np.ones(phoneme_1_indices.size),np.zeros(phoneme_2_indices.size))

# compute accuracy manually, (I wanted to use sklearn's function accuracy_score but was worried that marker may not have the library installled)
counter=0
for i in range(preds.size):
    if (y_truth[i]==preds[i]):
        counter+=1

accuracy = counter/preds.size
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()