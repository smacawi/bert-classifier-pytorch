from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import os
import pandas as pd
import torch

def read_data(data_path, paths):
    data = []
    counter = 0
    for path in paths:
        DIR = f"{data_path}/{path}"
        for filename in os.listdir(DIR):
            if filename.endswith(".csv") and all(x not in filename for x in ["train", "test"]):
                print(os.path.join(DIR, filename))
                with open(os.path.join(DIR, filename), 'r', encoding='utf8') as file:
                    df = pd.read_csv(file)
                    data.extend(list(zip(df["tweet_text"], df["choose_one_category"])))
                    counter += len(df["tweet_text"])
        print(f"Processed {counter} files.")
    print(f"Processed a total of {counter} files.")
    return(data)

def process_data(data, max_len, tokenizer):
    texts_raw = [tpl[0] for tpl in data]
    labels = [tpl[1] for tpl in data]
    print('Tokenizing sequences.')
    
    max_len_minus_two = max_len - 2
    texts = [tokenizer.tokenize(txt)[:max_len_minus_two] for txt in texts_raw]
    print('Max sentence length: ', max([len(txt) for txt in texts]))
    input_ids = []

    # For every sentence...
    for txt in texts:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_txt = tokenizer.encode(
                            txt[:max_len_minus_two], # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )   
        # Add the encoded sentence to the list.
        input_ids.append(encoded_txt)

    print('\nPadding/truncating all sentences to %d values...' % max_len)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", 
                              value=0, truncating="post", padding="post")

    print('\nDone.')
    
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
  
    return(input_ids, labels, attention_masks)

def to_torch_tensor(input_ids, labels, attention_masks):
    # Convert all inputs and labels into torch tensors, the required datatype 
    # for our model.
    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=2018, test_size=0)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                 random_state=2018, test_size=0)
    return(train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks)

def create_dataloader(inputs, labels, masks, batch_size = 4):

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.
    batch_size = batch_size

    # Create the DataLoader for our training set.
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return(dataloader)

