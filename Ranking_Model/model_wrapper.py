"""
ref
https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/text-transformers.ipynb#scrollTo=ddfafe98
https://pytorch-lightning.readthedocs.io/en/latest/
"""
import logging
from typing import List, Any
import math

import torch
import torch.nn.functional as F
from sklearn import metrics
from pytorch_lightning import LightningModule

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    ElectraTokenizer, ElectraConfig,
    AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
)
from models.ELECTRA_Model import ElectraForSequenceClassification, ElectraForSequenceClassificationCL


class Ranking_Wrapper_Model(LightningModule):
    def __init__(
        self,
        PTM_name_or_path,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_proportion: int = 0,
        weight_decay: float = 0.0,
        train_batch_size_pre_device: int = 32,
        args_str: str = '',
        problem_type = '',  
        cosine_schedule = False,
        contrastive_learning = None,
        transfer_learning = None,
        MCE_label_num = None,
        experiment = '',
        **kwargs
    ):
        '''
        problem type ['regression', 'classification_BCE', 'classification_MCE'],
        '''
        super().__init__()

        self.save_hyperparameters()

        if 'albert' in PTM_name_or_path.lower():
            Toknizer = AlbertTokenizer
            Model = AlbertForSequenceClassification
            Config = AlbertConfig
        elif 'electra' in PTM_name_or_path.lower():
            Toknizer = ElectraTokenizer
            Config = ElectraConfig
            if self.hparams.contrastive_learning:
                Model = ElectraForSequenceClassificationCL
            else:
                Model = ElectraForSequenceClassification
        else:
            logging.error('model name should in PTM_name_or_path')

        self.config = Config.from_pretrained(PTM_name_or_path)
        if self.hparams.transfer_learning == "csqa":
            self.config.problem_type = 'csqa'
            self.config.num_labels = 1
        elif self.hparams.problem_type == 'regression':
            self.config.problem_type = 'regression'
            self.config.num_labels = 1
        elif self.hparams.problem_type == "classification_BCE":
            self.config.problem_type = "multi_label_classification"
            self.config.num_labels = 1
        elif self.hparams.problem_type == "classification_MCE":
            self.config.problem_type = "single_label_classification"
            self.config.num_labels = self.hparams.MCE_label_num
        
        self.model = Model.from_pretrained(PTM_name_or_path, config=self.config)
        self.tokenizer = Toknizer.from_pretrained(PTM_name_or_path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, **inputs):
        # inference & prediction
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        if self.hparams.problem_type == "classification_BCE" \
            and not self.hparams.transfer_learning:
            batch['labels'] = batch['labels'].unsqueeze(1)
        if 'list_wise' in self.hparams.experiment:
            # import pdb; pdb.set_trace()
            labels = batch['labels']
            outputs = self(**batch)
            logits = outputs['logits']
            loss = self.listwise_loss(logits, labels)
            self.log("train_loss", loss, logger=True)
        else:
            outputs = self(**batch)
            loss = outputs[0]
            self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.hparams.problem_type == "classification_BCE" \
            and not self.hparams.transfer_learning:
            batch['labels'] = batch['labels'].unsqueeze(1)
        example_ids = batch.pop('example_id')
        if 'list_wise' in self.hparams.experiment:
            labels = batch.pop('labels')
            outputs = self(**batch)
            batch['labels'] = labels
            logits = outputs['logits']
            val_loss = self.listwise_loss(logits, labels)
        else:
            outputs = self(**batch)
            val_loss, logits = outputs[:2]

        if self.hparams.transfer_learning == 'csqa':
            predicts = torch.argmax(logits, 1)
        elif self.hparams.problem_type == 'regression':
            predicts = logits.squeeze(1)
        elif self.hparams.problem_type == 'classification_MCE':
            predicts = torch.argmax(logits, 1)
        elif self.hparams.problem_type == 'classification_BCE':
            predicts = torch.sigmoid(logits).squeeze()
            batch['labels'] = batch['labels'].squeeze()

        if 'list_wise' in self.hparams.experiment:
            predicts, batch, example_ids = self.listwise_dev_flatten(predicts, batch, example_ids)

        return {'loss': val_loss, 'predicts': predicts, 'labels': batch['labels'], 'example_ids': example_ids}

    def validation_epoch_end(self, validation_step_outputs):
        example_ids = []
        for example in validation_step_outputs:
            example_ids.extend(example['example_ids'])
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        if len(validation_step_outputs) > 1:
            last = validation_step_outputs.pop(-1) # 最后一个长度不一
        else:
            last = None
        predicts = validation_step_outputs[0]['predicts']
        labels = validation_step_outputs[0]['labels']
        for x in validation_step_outputs:
            predicts = torch.cat((predicts, x['predicts']))     
            labels = torch.cat((labels, x['labels']))       
        if last:
            predicts = torch.cat((predicts, last['predicts']), 0)
            labels = torch.cat((labels, last['labels']), 0)

        if self.hparams.transfer_learning == "csqa":
            acc = metrics.accuracy_score(labels.cpu(), predicts.cpu())
            self.log("acc", acc, prog_bar=True)
        elif self.hparams.problem_type == "classification_MCE":
            acc = metrics.accuracy_score(labels.cpu(), predicts.cpu())
            self.log("acc", acc, prog_bar=True)
        elif self.hparams.problem_type == "regression":
            ndcg1 = get_ndcg(example_ids, predicts.tolist(), labels.tolist(), 1)
            self.log("ndcg@1", ndcg1, prog_bar=True)
            ndcg3 = get_ndcg(example_ids, predicts.tolist(), labels.tolist(), 3)
            self.log("ndcg@3", ndcg3, prog_bar=True)
            logging.info(f"at setp {self.global_step} - vall_loss:{round(loss.item(), 5)}; ndcg@1:{ndcg1}; ndcg@3:{ndcg3}")
        else:
            labels_b = labels > 0.5
            auc = metrics.roc_auc_score(labels_b.cpu(), predicts.cpu())
            self.log("auc", auc, prog_bar=True)
            ndcg1 = get_ndcg(example_ids, predicts.tolist(), labels.tolist(), 1)
            self.log("ndcg@1", ndcg1, prog_bar=True)
            ndcg3 = get_ndcg(example_ids, predicts.tolist(), labels.tolist(), 3)
            self.log("ndcg@3", ndcg3, prog_bar=True)
            logging.info(f"at setp {self.global_step} - vall_loss: {round(loss.item(), 5)}; auc: {round(auc, 4)} ndcg@1: {ndcg1}; ndcg@3: {ndcg3}")

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # import pdb; pdb.set_trace()
        outputs = self(**batch)
        if self.hparams.problem_type == 'regression':
            score = outputs[0].squeeze(1)
        elif self.hparams.problem_type == "classification_CE":
            score = F.softmax(outputs[0], dim=1)[:,1]
        elif self.hparams.problem_type == "classification_BCE":
            score = torch.sigmoid(outputs[0]).squeeze()
        elif self.hparams.problem_type == "classification_MCE":
            score = F.softmax(outputs[0], dim=1)[:]
        return score

    # def predict_epoch_end(self, results: List[Any]):
    #     score_list = results[0]
    #     score = torch.flatten(torch.stack(score_list[:-1]))
    #     score = torch.cat((score, score_list[-1]))
    #     return score

    def set_example_num(self, example_num):
        self.example_num = example_num

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # import pdb; pdb.set_trace()
        # Calculate total steps
        if isinstance(self.trainer.gpus, list):
            # specific device like [6], [6,7]
            gpus = len(self.trainer.gpus)
        else:
            gpus = self.trainer.gpus
        
        tb_size = self.hparams.train_batch_size_pre_device * max(1, gpus)
        ab_size = self.trainer.accumulate_grad_batches
        total_steps = int((self.example_num * int(self.trainer.max_epochs) // tb_size) // ab_size )
        warmup_steps = int(self.hparams.warmup_proportion * total_steps)
        logging.info(f'total_steps: {total_steps}; warmup_steps: {warmup_steps}')

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        
        if self.hparams.get('cosine_schedule', False):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def listwise_dev_flatten(predicts, batch, example_ids):
        predicts_new = []
        labels_new = []
        example_ids_new = []
        batch_size = predicts.shape[0]
        labels = batch['labels']
        for i in range(batch_size):
            predict = predicts[i]
            label = labels[i]
            j = 0
            for l in label:
                if l == -1:
                    break
                else:
                    j += 1
            predicts_new.append(predict[:j])
            labels_new.append(label[:j])
            example_ids_new.extend([example_ids[i]] * j)
        predicts = predicts_new[0]
        labels = labels_new[0]
        for predict, label in zip(predicts_new[1:], labels_new[1:]):
            predicts = torch.cat((predicts, predict))
            labels = torch.cat((labels, label))
        batch['labels'] = labels
        return predicts, batch, example_ids_new

    @staticmethod
    def listwise_loss(logits, labels):
        # labels
        total_loss = 0
        batch_size = logits.shape[0]
        for i in range(batch_size):
            logit = logits[i]
            label = labels[i]
            j = 0
            for l in label:
                if l == -1:
                    break
                else:
                    j += 1
            logit = logit[:j]
            label = label[:j]
            P_z_i = F.softmax(logit, dim=0)
            P_y_i = F.softmax(label, dim=0)
            total_loss += - torch.sum(P_y_i * torch.log(P_z_i))
        return total_loss / batch_size

#### metric

def cal_dcg(list_label, list_rank_score, top_k):
    rank_idx = sorted(range(len(list_rank_score)), key=lambda k: list_rank_score[k], reverse=True)
    dcg = 0
    max_rank = 0
    if top_k == -1:
        max_rank = len(list_rank_score) + 1
    else:
        max_rank = min(top_k + 1, len(list_rank_score) + 1)
    for i in range(1, max_rank):
        dcg += (2 ** (list_label[rank_idx[i - 1]]) - 1) / float(math.log(i + 1, 2))
    return dcg

def cal_ndcg(list_label, list_rank_score, top_k):
    dcg_predict = cal_dcg(list_label, list_rank_score, top_k)
    dcg_label = cal_dcg(list_label, list_label, top_k)
    return float(dcg_predict)/(dcg_label + 0.00000000000000001) # avoid div zero
#对每次搜索计算ndcg，然后取平均

def get_ndcg(query_id_list, score_list, label_list, top_k):
    dict_query_id2url_list = dict()
    #对数据按照query_id进行group
    for i in range(len(query_id_list)):
        query_id = query_id_list[i]
        score = score_list[i]
        label = label_list[i]
        if query_id not in dict_query_id2url_list:
            dict_query_id2url_list[query_id] = list()
        dict_query_id2url_list[query_id].append((score, label))
    ndcg_sum = 0
    q_cnt = 0
    #计算每个query_id下的ndcg
    for query_id in dict_query_id2url_list:
        score_label_tuple_list = dict_query_id2url_list[query_id]
        tmp_score_list = list()
        tmp_label_list = list()
        for i in range(len(score_label_tuple_list)):
            tmp_score_list.append(float(score_label_tuple_list[i][0]))
            tmp_label_list.append(float(score_label_tuple_list[i][1]))
            #只对存在label大于0的query_id进行计算
        if sum(tmp_label_list) > 0:
            ndcg = cal_ndcg(tmp_label_list, tmp_score_list, top_k)
            ndcg_sum += ndcg
        q_cnt += 1
    ndcg_avg = ndcg_sum / q_cnt
    ndcg_avg = round(ndcg_avg, 4)
    return ndcg_avg