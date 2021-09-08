import os
import ast
import argparse
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='uda')
    args = parser.parse_args()
    return args
    
'''
1. https://github.com/SanghunYun/UDA_pytorch 의 data.zip을 사용하는 경우
    (1) git clone https://github.com/SanghunYun/UDA_pytorch
    (2) cd UDA_pytorch
    (3) unzip data.zip
    (4) data 폴더를 원하는 {data_folder}에 옮겨 놓기

2. IMDB 데이터를 그대로 받아와서 사용하는 경우
    (1) 데이터 다운로드
    (2) 데이터 전처리 (html 태그 제거)
    (3) supervised | unsupervised data 나누기
    
'''

def from_uda_data(data_folder):
    valid_format = ['imdb_sup_test.txt', 'imdb_sup_train.txt', 'imdb_unsup_train.txt']
    assert valid_format == os.listdir(data_folder), \
        print(f"IMPORT ERROR : {data_folder} does not contain a valid data files: {valid_format}")

    sup_train = pd.read_csv(data_folder+'/imdb_sup_train.txt', delimiter='\t') # shape (20, 4)
    sup_train = sup_train.to_dict()
    for k,v in sup_train.items():
        if isinstance(v,dict):
            sup_train[k] = list(v.values())
        else:
            raise NotImplementedError
    sup_test = pd.read_csv(data_folder+'/imdb_sup_test.txt', delimiter='\t') # shape (25000,4)
    sup_test = sup_test.to_dict()
    for k,v in sup_test.items():
        if isinstance(v,dict):
            # sup_test[k] = torch.tensor( [ast.literal_eval(v[i]) if isinstance(v[i],str) else v[i] for i in v], dtype=torch.long)
            sup_test[k] = list(v.values())
        else:
            raise NotImplementedError
    
    unsup_train = pd.read_csv(data_folder+'/imdb_unsup_train.txt', delimiter='\t') # shape (69972, 6)
    unsup_train = unsup_train.to_dict()
    
    for k,v in unsup_train.items():
        if isinstance(v,dict):
            # unsup_train[k] = torch.tensor( [ast.literal_eval(v[i]) if isinstance(v[i],str) else v[i] for i in v], dtype=torch.long)
            unsup_train[k] = list(v.values())
        else:
            raise NotImplementedError
    
    return sup_train, sup_test, unsup_train


def _from_imdb_data(data_folder, args, sampling=None, n_label=5000, n_unlabel=5000):

    # data = load_data(args, 100)
    with open(args.data_id_path) as f:
        id_list = f.read().split()
    if isinstance(sampling, int):
        id_list = id_list[:sampling]

    data = []
    for idx in id_list:
        sent, file_name = idx.split('_', 1)
        with open('/'.join([args.data_folder, sent, file_name]), encoding='utf-8' ) as f:
            data.append( [f.read(), sent] )
    
    # L_data, U_data = split_data(data, 50, 50)
    assert len(data) >= n_label + n_unlabel
    L_data = data[:n_label]
    U_data = data[n_label: n_label + n_unlabel + 1]

    ## TO BE IMPLEMENTED ##
    # augmented_data = augment_data(U_data)
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    # Access the underlying TransformerModel
    assert isinstance(en2de.models[0], torch.nn.Module)
    # Translate from En-De
    de = en2de.translate('PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.')


class UDADataset(Dataset):
    def __init__(self, data):
        '''Customized dataset for UDA (un)labeled dataset
        -- INPUT
        labeled data : {'input_ids', 'input_mask', 'input_type_ids', 'label_ids'}
        unlabeled data : {'ori_input_ids', 'ori_input_mask', 'ori_input_type_ids', 'aug_input_ids', 'aug_input_mask', 'aug_input_type_ids'}
        
        -- getitem
        return (original text, augmented text, label)
        -> (text, _, label) for labeled data
        -> (text, augmented text, _) for unlabeled data
        '''
        if 'input_ids' in data.keys():
            self.text = data[ 'input_ids' ] 
            self.label = data['label_ids']
            self.aug = self.text
        
        elif 'aug_input_ids' in data.keys():
            self.text = data[ 'ori_input_ids' ] 
            self.aug  = data[ 'aug_input_ids' ]
            self.label = [0]*len(self.text)

        else:
            raise ValueError

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = torch.tensor( ast.literal_eval(self.text[idx]) , dtype=torch.long)
        aug_text = torch.tensor( ast.literal_eval(self.aug[idx]) , dtype=torch.long)
        label = torch.tensor( ast.literal_eval(self.label[idx]) , dtype=torch.long)
        return text, aug_text, label
    
