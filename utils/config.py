class Config():
    def __init__(self):
        self.data = 'uda'
        self.uda_data_folder = 'data'
        self.imdb_data_folder = 'data/aclImdb/train' # train > pos | neg | unsup > {id:int}.txt
        self.data_id_path = '../uda/text/data/IMDB_raw/train_id_list.txt'

class TrainingConfig():
    def __init__(self):
        self.data = 'uda'
        self.do_train = True
        self.do_eval = True
        self.batch_size = 8
        self.weight_decay = 0.1
        self.learning_rate = 0.1
        self.num_train_steps=1000
        self.num_train_epochs=10
        self.num_labels = 2
        self.num_warmup_steps = 500
        self.gradient_accumulation_steps = 1
        self.max_train_steps=1000


class ModelConfig():
    def __init__(self):
        self.model_name_or_path = 'bert-base-cased'
        self.config_name = None
        self.tokenizer_name = None
        self.cache_dir = None
        self.use_auth_token = None

