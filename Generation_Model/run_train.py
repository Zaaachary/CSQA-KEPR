import argparse
import os
import logging
from statistics import mode
import time
from collections import OrderedDict
import torch
from functools import partial

# import pdb; pdb.set_trace()
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
import numpy as np
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import general_eval, evaluate
from protoqa_evaluator.scoring import wordnet_score

from model_wrapper import Generation_Model
from data import ProtoQA_Dataset, CommonGen_Dataset
from run_infer import run_beam_search, dump_data

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger('')

max_answers = {
    f"Max_Answers_{k}": partial(general_eval, max_pred_answers=k)
    for k in [1, 3, 5, 10, None]
    # for k in [1, 3, 5, 10]
}
max_incorrect = {
    f"Max_Incorrect_{k}": partial(general_eval, max_incorrect=k)
    for k in [1, 3, 5]
    # for k in [1, 3, 5]
}
exact_match_all_eval_funcs = {**max_incorrect, **max_answers}
wordnet_all_eval_funcs = {
    f"WordNet_{k}": partial(v, score_func=wordnet_score, score_matrix_transformation=np.round)
    for k, v in exact_match_all_eval_funcs.items()
}
all_eval_funcs = {}
all_eval_funcs.update(wordnet_all_eval_funcs)

def gpus_parser(gpus):
    """
    match input type with pytorch-lightning type
    "6,7" -> [6, 7]; "6" -> [6,]; "-1" -> 0
    """
    accelerator = None
    if gpus == "-1":    # no cpu
        gpus = 0
    elif "," in gpus:   # muliti gpu
        gpus = gpus.split(",")
        if "" in gpus:
            gpus.remove("")
        gpus = list(map(int, gpus))
        accelerator = "ddp"
    else:               # single gpu
        gpus = [int(gpus),]
    return gpus, accelerator

def get_version_name(args):
    version_name = ''
    gpus, _ = gpus_parser(args.gpus)
    gpu_num = len(gpus) if isinstance(gpus, list) else 1
    version_name += f"bz={gpu_num}x{args.train_batch_size_per_gpu}x{args.gradient_accumulation_step}"
    version_name += f"_ep={args.epoch}_lr={args.learning_rate}_ae={args.adam_epsilon}_seed={args.seed}"
    if args.experiment:
        version_name += f"_{args.experiment}"
    return version_name

def set_logger(args, version_name):
    tblogger = TensorBoardLogger(
        args.output_path,
        name=args.task_name,
        version=version_name
    )
    root = os.path.join(args.output_path, args.task_name)
    if not os.path.exists(root):
        os.mkdir(root)
    root = os.path.join(root, version_name)
    if not os.path.exists(root):
        os.mkdir(root)
    
    handler = logging.FileHandler(os.path.join(root, f'train.{time.strftime("%Y-%m-%d.%H:%M:%S")}.log'))
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s', datefmt='%y/%m/%d %H:%M')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return tblogger, root

def eval(args, output_root, gpus, tokenizer, model):
    output_root = os.path.join(output_root, 'evaluate')
    output_file = os.path.join(output_root, 'lastmodel.jsonl')
    try:
        os.mkdir(output_root)
    except:
        logger.info('evaluate root exists')
    device = gpus[0]
    device = f"cuda:{device}"
    model.to(device)
    model.eval()
    
    dataset = ProtoQA_Dataset(
        args.eval_target, args.model_type, None, None, 
        True, tokenizer=tokenizer, dataset_type='predict', evaluate=True,
        experiment=args.experiment, wkdt_path=args.wkdt_path
    )
    all_result_list, all_info_lsit = run_beam_search(args, dataset, tokenizer, 30, 12, model, device)
    
    logger.info(f'save result to < {output_root} >')
    dump_data(all_result_list, output_file, mode='jsonl')
    dump_data(all_info_lsit, output_file.replace('jsonl', '')+'info.json', mode='json')
    
    logger.info(f'compute wordnet score')
    answer_dict = OrderedDict()
    for line in all_result_list:
        answer_dict.update(line)
    question_data = load_question_answer_clusters_from_jsonl(args.eval_target)
    result_dict = {}
    for eval_name, eval_func in all_eval_funcs.items():
        score = evaluate(eval_func, question_data, answers_dict=answer_dict)
        score = np.mean([x.score for x in score.values()])
        result_dict[eval_name] = round(score, 4)
        logger.info(f'{eval_name}: {score}')
    dump_data(result_dict, os.path.join(output_root, 'wordnet_score.json'), mode='json')
      
def main(args):
    # import pdb; pdb.set_trace()
    gpus, accelerator = gpus_parser(args.gpus)
    version = get_version_name(args)
    tblogger, output_root = set_logger(args, version)
    logger.info(str(args))

    seed_everything(args.seed)
    # GPT2 model & tokenizer init
    model = Generation_Model(
        PTM_name_or_path=args.PTM_name_or_path,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        train_batch_size_pre_device=args.train_batch_size_per_gpu,
        args_str=str(args),
        model_type=args.model_type,
        T5_split=args.T5_split,
        experiment=args.experiment
    )
    tokenizer = model.get_tokenizer()
    # dataset -> dataloader
    if args.commongen:
        train_dataset = CommonGen_Dataset(
            args.dataset_path, max_src_len=args.max_src_len, 
            max_tgt_len=args.max_tgt_len, tokenizer=tokenizer,
            dataset_type='train', experiment=args.experiment
        )
        dev_dataset = CommonGen_Dataset(
            args.dataset_path, max_src_len=args.max_src_len, 
            max_tgt_len=args.max_tgt_len, tokenizer=tokenizer,
            dataset_type='dev', experiment=args.experiment
        )
    else:
        train_dataset = ProtoQA_Dataset(
            args.dataset_path, args.model_type,
            args.max_src_len, args.max_tgt_len,
            args.overwrite_cache,
            tokenizer=tokenizer, dataset_type='train',
            experiment=args.experiment,
            wkdt_path=args.wkdt_path,
            seed=args.data_seed)
        dev_dataset = ProtoQA_Dataset(
            args.dataset_path, args.model_type,
            args.max_src_len, args.max_tgt_len,
            args.overwrite_cache,
            experiment=args.experiment,
            tokenizer=tokenizer, dataset_type='dev',
            wkdt_path=args.wkdt_path)
    
    train_dataloader = train_dataset.make_dataloader(
        batch_size=args.train_batch_size_per_gpu)
    model.set_example_num(len(train_dataset))
    dev_dataloader = dev_dataset.make_dataloader(
        batch_size=args.dev_batch_size_per_gpu)

    # set trainer
    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        every_n_epochs=1,
        filename="{epoch:02d}-{step}-{val_loss:.6f}",
        save_top_k=2,
        mode="min",
        save_weights_only=True  # params only, without optimizer state
    )

    # from epoch a, it starts accumulating every b batches. Here we have a-1 instead of a   {a-1: b}
    accumulator = GradientAccumulationScheduler(
        scheduling={0: args.gradient_accumulation_step})

    trainer = Trainer(
        max_epochs=args.epoch,
        val_check_interval=0.1,
        gpus=gpus,
        accelerator=accelerator,
        # fast_dev_run=True,  # enable when debug
        deterministic=True,
        default_root_dir=args.output_path,
        logger=tblogger,
        precision=16 if args.fp16 else 32,
        callbacks=[checkpoint_callback, accumulator]
    )
    # trainer.fit(model, train_dataloader)
    trainer.fit(model, train_dataloader, dev_dataloader)
    # protoqa eval after training
    if args.do_eval and not args.commongen:
        eval(args, output_root, gpus, tokenizer, model)
    logger.info('finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model path and name
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default='gpt2')
    parser.add_argument("--wkdt_path", type=str, default='')
    parser.add_argument("--task_name", type=str, required=True,
                        help='feature in input/model level')

    # dataset
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--max_src_len", type=int, default=None,
        help='Bart source_length, GPT2 seq_len')
    parser.add_argument("--max_tgt_len", type=int, default=None,
        help='Bart target')
    parser.add_argument("--experiment", type=str, default='')
    parser.add_argument("--commongen", action='store_true', default=False)

    # hparams & device
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--train_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--gradient_accumulation_step", default=1, type=int)
    parser.add_argument("--dev_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data_seed", default=42, type=int)
    parser.add_argument("--T5_split", default=None, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gpus", type=str,
                        help="-1:not use; x:card id; [6,7]: use card 6 and 7")
    
    # eval
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--eval_target", type=str, default='')
    parser.add_argument("--eval_length", type=int, default=188)


    args = parser.parse_args()
    main(args)
