import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.data_folder = data_folder
        
        # Initialize tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Load and process data
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        """Process the data from files"""
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        # Load SQL queries (not available for test set)
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
        else:
            sql_queries = None
        
        # Create data list
        data = []
        for i, nl_query in enumerate(nl_queries):
            item = {
                'nl_query': nl_query,
                'sql_query': sql_queries[i] if sql_queries else None,
                'idx': i
            }
            data.append(item)
        
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input (natural language)
        encoder_input = self.tokenizer(
            item['nl_query'],
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        if self.split != 'test':
            # Tokenize output (SQL query)
            decoder_target = self.tokenizer(
                item['sql_query'],
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            
            return {
                'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
                'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
                'decoder_input_ids': decoder_target['input_ids'].squeeze(0),
                'nl_query': item['nl_query'],
                'sql_query': item['sql_query']
            }
        else:
            # Test set - no target SQL
            return {
                'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
                'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
                'nl_query': item['nl_query']
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract components
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    # For decoder: we need to create input and target sequences
    # Decoder input: shift right (add pad token at the beginning)
    # Decoder target: the original sequence
    decoder_targets_list = []
    decoder_inputs_list = []
    initial_decoder_inputs_list = []
    
    for decoder_ids in decoder_input_ids:
        # Target is the full sequence
        decoder_targets_list.append(decoder_ids)
        
        # Input is shifted right - prepend with pad token (T5 uses pad_token as decoder_start_token)
        decoder_input = torch.cat([torch.tensor([PAD_IDX]), decoder_ids[:-1]])
        decoder_inputs_list.append(decoder_input)
        
        # Initial decoder input is just the start token
        initial_decoder_inputs_list.append(torch.tensor([PAD_IDX]))
    
    # Pad decoder sequences
    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs_list)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract components
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    # Initial decoder inputs (start token for generation)
    initial_decoder_inputs = torch.tensor([[PAD_IDX]] * len(batch))
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    """Load data for prompting experiments"""
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x