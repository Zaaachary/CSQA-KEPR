import argparse
import os
import logging
import pdb
import time
from importlib_metadata import version

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model_wrapper import Ranking_Wrapper_Model
from data import CSQA_Dataset, Ranking_Dataset

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger('')

def gpus_parser(gpus):
    """
    match input type with pytorch-lightning type
    "6,7" -> [6, 7]; "6" -> [6,]; "-1" -> 0
    """
    accelerator = None
    if gpus == "-1":
        # no cpu
        gpus = 0
    elif "," in gpus:
        # muliti gpu
        gpus = gpus.split(",")
        if "" in gpus:
            gpus.remove("")
        gpus = list(map(int, gpus))
        accelerator = "ddp"
    else:
        # single gpu
        gpus = [int(gpus),]
    return gpus, accelerator

def get_version_name(args):
    version_name = ''
    gpus, _ = gpus_parser(args.gpus)
    gpu_num = len(gpus) if isinstance(gpus, list) else 1
    temp=os.path.basename(args.dataset_path)
    device = torch.cuda.get_device_name(0)[-8:].replace(' ','')
    version_name += f"data={temp}"
    version_name += f"_bz={gpu_num}x{args.train_batch_size_per_gpu}x{args.gradient_accumulation_step}"
    version_name += f"_ep={args.epoch}_lr={args.learning_rate}_ae={args.adam_epsilon}_seed={args.seed}"
    version_name += f"_{device}"
    if args.cosine_schedule:
        version_name += "_cos"
    if args.add_wkdt:
        version_name += "_wkdt"
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
    return tblogger

def load_dataset(args, tokenizer):
    if args.transfer_learning == "csqa":
        train_dataset = CSQA_Dataset(
            args.dataset_path, args.max_len, tokenizer, dataset_type='train',
            problem_type=args.problem_type
        )
        dev_dataset = CSQA_Dataset(
            args.dataset_path, args.max_len, tokenizer, dataset_type='dev',
            problem_type=args.problem_type
        )
    else:
        train_dataset = Ranking_Dataset(
            args.dataset_path, args.max_len, tokenizer, "train", args.overwrite_cache,
            problem_type = args.problem_type,
            add_wkdt = args.add_wkdt,
            wkdt_path = args.wkdt_path,
            contrastive_learning = args.contrastive_learning,
            experiment = args.experiment,
            keyword_path=args.keyword_path
        )

        dev_dataset = Ranking_Dataset(
            args.dataset_path, args.max_len, tokenizer, "dev", args.overwrite_cache,
            problem_type = args.problem_type,
            add_wkdt = args.add_wkdt,
            wkdt_path = args.wkdt_path,
            experiment = args.experiment,
            keyword_path=args.keyword_path
        )
    return train_dataset, dev_dataset

def main(args):
    seed_everything(args.seed)
    version = get_version_name(args)
    tblogger = set_logger(args, version)
    logger.info(str(args))

    # model & tokenizer init
    model = Ranking_Wrapper_Model(
        PTM_name_or_path=args.PTM_name_or_path,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        train_batch_size_pre_device=args.train_batch_size_per_gpu,
        args_str=str(args),
        problem_type=args.problem_type,
        cosine_schedule=args.cosine_schedule,
        contrastive_learning=args.contrastive_learning,
        transfer_learning=args.transfer_learning,
        MCE_label_num=args.MCE_label_num,
        experiment=args.experiment
    )
    if args.transfer_ckpt_path:
        checkpoint = torch.load(args.transfer_ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    tokenizer = model.get_tokenizer()

    # dataset -> dataloader
    logger.info(f"load dataset from {args.dataset_path}")
    train_dataset, dev_dataset = load_dataset(args, tokenizer)
    train_dataloader = train_dataset.make_dataloader(
        batch_size=args.train_batch_size_per_gpu
    )
    dev_dataloader = dev_dataset.make_dataloader(
        batch_size=args.dev_batch_size_per_gpu)
    model.set_example_num(len(train_dataset))    # for setting schedule and optimizer; need extra line for this, thanks to the pytorchlightning :(
    # set trainer
    if args.transfer_learning == 'csqa' or args.problem_type == "classification_MCE":
        monitor = 'acc'
        filename = "{epoch:02d}-{step}-{val_loss:.4f}-{acc:.4f}"
    elif args.problem_type == 'regression':
        monitor = 'val_loss'
        filename = "{epoch:02d}-{step}-{val_loss:.4f}"
    else:
        monitor = 'auc'
        filename="{epoch:02d}-{step}-{val_loss:.4f}-{auc:.4f}"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        every_n_epochs=1,
        filename=filename,
        save_top_k=4,
        mode="min" if args.problem_type == 'regression' else "max",
        # save_last=True,
        save_weights_only=True  # params only, without optimizer state
    )

    gpus, accelerator = gpus_parser(args.gpus)
    trainer = Trainer(
        max_epochs=args.epoch,
        # val_check_interval=0.25,
        val_check_interval=0.1,
        # check_val_every_n_epoch=1,
        gpus=gpus,
        # strategy=accelerator,
        accelerator=accelerator,
        # fast_dev_run=True,  # enable when debug
        deterministic=True,
        default_root_dir=args.output_path,
        logger=tblogger,
        precision=16 if args.fp16 else 32,
        accumulate_grad_batches=args.gradient_accumulation_step,
        callbacks=[checkpoint_callback, ],
    )

    if args.lr_tuning:
        lr_finder = trainer.tuner.lr_find(model, train_dataloader, dev_dataloader)
        print(lr_finder.results)
        new_lr = lr_finder.suggestion()
        print(new_lr)
        fig = lr_finder.plot(suggest=True)
        fig.savefig(args.output_path + '/tune.png')
    else:
        # trainer.fit(model, train_dataloader)
        trainer.fit(model, train_dataloader, dev_dataloader)
    logger.info('finished')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # path and name
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--wkdt_path", type=str, default=None)
    parser.add_argument("--transfer_ckpt_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--keyword_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True,
                        help='feature in input/model level')

    # dataset & model mode
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--problem_type", type=str, 
        choices=['regression', 'classification_CE', 'classification_BCE', 'classification_MCE'])
    parser.add_argument("--MCE_label_num", type=int, default=0)
    parser.add_argument("--add_wkdt", action='store_true', default=False)
    parser.add_argument("--transfer_learning", choices=[None, 'csqa'], default=None)

    # hparams & device
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_proportion", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    
    parser.add_argument("--cosine_schedule", action='store_true', default=False)
    parser.add_argument("--contrastive_learning", choices=[None, 'triple'], default=None)
    parser.add_argument("--train_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--gradient_accumulation_step", default=1, type=int)
    parser.add_argument("--dev_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr_tuning", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--experiment", type=str, default='')
    parser.add_argument("--gpus", type=str,
                        help="-1:not use; x:card id; [6,7]: use card 6 and 7")

    args = parser.parse_args()

    main(args)
