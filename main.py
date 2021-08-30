import os
import torch

class Config():
    def __init__(self):
        self.data_folder = 'data/aclImdb/train' # train > pos | neg | unsup > {id:int}.txt
        self.data_id_path = '../uda/text/data/IMDB_raw/train_id_list.txt'

def load_data(args, sampling=None):
    # open id file
    with open(args.data_id_path) as f:
        id_list = f.read().split()
    if isinstance(sampling, int):
        id_list = id_list[:sampling]

    
    # load corresponding data
    data = []
    for idx in id_list:
        sent, file = idx.split('_', 1)
        with open('/'.join([args.data_folder, sent, file]), encoding='utf-8' ) as f:
            data.append( [f.read(), sent] )
    return data

def split_data(data, n_label=5000, n_unlabel=5000):
    # return Labeled data and Unlabeled Data
    assert len(data) >= n_label + n_unlabel
    L_data = data[:n_label]
    U_data = data[n_label: n_label + n_unlabel + 1]
    return L_data, U_data


def augment_data(data):
    # import model and back translate
    import pdb; pdb.set_trace()
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='gpt2')
    # 그나마 gpt2..
    # Access the underlying TransformerModel
    assert isinstance(en2de.models[0], torch.nn.Module)

    # Translate from En-De
    de = en2de.translate('PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.')


        
def main(args):
    data = load_data(args, 100)
    L_data, U_data = split_data(data, 50, 50)
    augmented_data = augment_data(U_data)


if __name__ == '__main__':
    args = Config()
    main(args)
    # from IPython import embed; embed()