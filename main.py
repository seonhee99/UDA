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
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
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


    model_args = ModelConfig()
    logging.info(f'\tMODELLING {model_args.model_name_or_path} model...')

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_config(config)


    ## TODO : training까지
    logger.info('\tTRAIN')
    # training_args = TrainingArguments("test_trainer")
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    # )

    train_dataset = dp.UDADataset(train)
    test_dataset = dp.UDADataset(test)
    unlabeled_dataset = dp.UDADataset(unlabeled)

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)
    unlabeled_loader = DataLoader(unlabeled_dataset)

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
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    # (optional)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer = optimizer
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.num_training_steps,
    )
    metric = load_metric('accuracy')
    # progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    logging.info('\t*** Running training & eval ***')
    for epoch in range(training_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = output.loss
            loss /= training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

    ## 하단 : Trainer 기준 코드 (수정필요)
    if training_args.do_train and training_args.do_eval:
        best_acc = 0
        for _ in range(0, training_args.num_train_steps, save_checkpoints_steps):
            trainer.train(train_input_fn, step)
            dev_res = trainer.evaluate(eval_fn, eval_step)
            for k in dev_res.keys():
                print(f"\t{k} : {dev_res[k]}")
                dev_res[k] = dev[k].item()


if __name__ == '__main__':
    main()