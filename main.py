import os
import math
import argparse
import logging
import tqdm

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_metric
from accelerate import Accelerator # pip install accelearte if necessary

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    # AutoTokenizer,
    # DataCollatorWithPadding,
    # EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    # set_seed,
)

from utils import data_preprocess as dp
from utils.config import *
# from utils import 
    
def main():
    cfg = Config()
    training_args = TrainingConfig()
    model_args = ModelConfig()

    logging.basicConfig(level=logging.INFO)
    logging.info('RUN main.py')

    logging.info('\tLOAD data...')
    if 'imdb' == cfg.data:
        data = dp.from_imdb_data(cfg.imdb_data_folder, cfg)
        raise NotImplementedError
    elif 'uda' == cfg.data:
        data = dp.from_uda_data(cfg.uda_data_folder) # sup_train, sup_test, unsup_train
        train,test,unlabeled = data
    else:
        raise NotImplementedError

    logging.info(torch.cuda.is_available())
    model_args = ModelConfig()
    logging.info(f'\tMODELLING {model_args.model_name_or_path} model...')

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_config(config)


    logger.info('\tTRAIN')
    # import pdb; pdb.set_trace()
    train_dataset = dp.UDADataset(train)
    test_dataset = dp.UDADataset(test)
    unlabeled_dataset = dp.UDADataset(unlabeled)

    train_len = math.ceil( len(train_dataset) / training_args.batch_size )
    unlabeled_batch_size = math.ceil(len(unlabeled_dataset) / train_len)

    train_loader = DataLoader(train_dataset, batch_size=training_args.batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=training_args.batch_size)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size)

    accelerator = Accelerator()
    logger.info(accelerator.state)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    # model, optimizer, train_loader, test_loader = accelerator.prepare(
    #     model, optimizer, train_loader
    # )
    # eval_dataloader = iter(test_loader) #### iter 왜 넣었지 ..? ###
    num_update_steps_per_epoch = math.ceil(len(train_loader) / training_args.gradient_accumulation_steps)
    # (optional)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.num_train_steps,
    )
    metric = load_metric('accuracy')
    # progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    CE_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    ## torch.nn.CrossEntropyLoss : LogSoftmax + NLL Loss
    KL_loss = torch.nn.KLDivLoss(reduction="batchmean")

    logging.info('\t*** Running training & eval ***')


    for epoch in range(training_args.num_train_epochs):
        
        model.train()
        print(f'==== EPOCH {epoch} training ====')
        loss = 0
        unlabled_iter = iter(unlabeled_dataloader)

        for step, batch in enumerate(train_loader):
            text, _, label = batch
            outputs = model(input_ids=text['input_ids'], attention_mask=text['input_mask'], token_type_ids=text['input_type_ids'])
            cross_entropy_loss = CE_loss(outputs[0], label)
            supervised_loss = cross_entropy_loss / training_args.gradient_accumulation_steps

            unlabeled_batch = next(unlabled_iter)
            u_text, a_text, _ = unlabeled_batch
            # from IPython import embed; embed()
            a_output = model(input_ids=a_text['aug_input_ids'], attention_mask=a_text['aug_input_mask'], token_type_ids=a_text['aug_input_type_ids'])[0]
            # a_output = model(a_text)[0]

            model.eval()
            with torch.no_grad():
                u_output = model(input_ids=u_text['ori_input_ids'], attention_mask=u_text['ori_input_mask'], token_type_ids=u_text['ori_input_type_ids'])[0]
                # u_output = model(u_text)[0]
            model.train()

            unsupervised_loss = KL_loss(u_output, a_output)

            loss = (supervised_loss + unsupervised_loss * consistency_loss) / training_args.gradient_accumulation_steps

            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

        # for step, batch in enumerate(unlabeled_dataloader):
        #     # unsupervised learning
        #     ori_text, aug_text, _ = batch
        #     output_aug = model(input_ids=aug_text['aug_input_ids'], attention_mask=aug_text['aug_input_mask'], token_type_ids=aug_text['aug_input_type_ids'])
        #     with torch.no_grad():
        #         output_ori = model(input_ids=ori_text['ori_input_ids'], attention_mask=ori_text['ori_input_mask'], token_type_ids=ori_text['ori_input_type_ids'])
            
        #     consistency_loss = KL_loss(output_ori[0], output_aug[0])
        #     loss = training_args.consistency_loss_weight * consistency_loss / training_args.gradient_accumulation_steps
        #     accelerator.backward(loss)
        #     #### loss 를 더해서 update해주어야 하진 않은지?
                
        #     if step % training_args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
        #         optimizer.step()
        #         lr_scheduler.step()
        #         optimizer.zero_grad()
        #         # progress_bar.update(1)
        #         completed_steps += 1

            if completed_steps >= training_args.max_train_steps:
                break

        print('model.eval() begins')
        model.eval()
        
        for step, batch in enumerate(test_loader):
            text, _, label = batch
            outputs = model(input_ids=text['input_ids'], attention_mask=text['input_mask'], token_type_ids=text['input_type_ids'])
            predictions = outputs[0].argmax(dim=-1) # if not is_regression else outputs.logits.squeeze()
            metric.add_batch( # int object is not iterable error
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(label),
            )

        eval_metric = metric.compute() #### ERROR ####
        '''TODO
        1. test set & train set 줄여서 500개씩만 돌려보기
        2. pdb
        3. try : except :  m
        --> eval이 iter 객체라 아마 한 에폭 끝나고 다음에 다시 for문 돌리니까 에러 날 것
        --> iter객체로 만들어주는 부분을 주석처리 해놧으므로 아마 다시 돌리면 잘 돌아갈 것 같습니당
        '''
        logger.info(f"epoch {epoch}: {eval_metric}")

if __name__ == '__main__':
    main()