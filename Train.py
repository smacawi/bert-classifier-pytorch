from DatasetReader import *
from Model import *
from transformers import BertTokenizer

import os
import pickle
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

DATA_PATH = "CrisisNLP_labeled_data_crowdflower"
PATHS = ["2014_California_Earthquake", "2014_Chile_Earthquake_en", "2013_Pakistan_eq",
               "2014_Hurricane_Odile_Mexico_en","2014_India_floods", "2014_Pakistan_floods",
               "2014_Philippines_Typhoon_Hagupit_en","2015_Cyclone_Pam_en","2015_Nepal_Earthquake_en"]

print("Processing training data.")
data_train = read_data(DATA_PATH, PATHS, "train")
data_train = data_train

print("Processing validation data.")
data_val = read_data(DATA_PATH, PATHS, "test")

max_len = max(
    len(max([tpl[0] for tpl in data_val], key = len)),
    len(max([tpl[0] for tpl in data_train], key = len)))
epochs = 1
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print("Tokenizing training data.")
train_inputs, train_labels, train_masks = process_data(data_train, max_len, tokenizer)

print("Tokenizing validation data.")
validation_inputs, validation_labels, validation_masks = process_data(data_val, max_len, tokenizer)

print("Creating dataloaders.")
train_dataloader = create_dataloader(train_inputs, train_labels, train_masks)
validation_dataloader = create_dataloader(validation_inputs, validation_labels, validation_masks)

print("Initializing model.")
model, scheduler, optimizer = BertCLS_initialize(train_dataloader, num_labels = 9, epochs = epochs)

print("Training model.")
torch.cuda.empty_cache()
model, loss_values, outputs = BertCLS_train(model, optimizer, scheduler, 
                                             train_dataloader, validation_dataloader, 
                                             device, epochs = epochs)

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
output_dir = './model_save_attention_1epoch/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

try:
    print("Saving models outputs and loss values to %s" % output_dir)
    pickle.dump(outputs, open(f"{output_dir}/outputs.pkl", "wb" ))
    pickle.dump(loss_values, open(f"{output_dir}/loss_values.pkl", "wb" ))
except:
    print("Failed to save outputs, check code for errors.")
