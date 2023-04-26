# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ELECTRA model. """
import imp
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, get_activation

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

from transformers.utils import logging
from transformers import ElectraPreTrainedModel, ElectraModel


logger = logging.get_logger(__name__)


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        # if self.config.num_labels == 1:
        #     x = get_activation("sigmoid")(x)
        # else:
        #     x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # import pdb; pdb.set_trace()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(input_ids.shape) == 3:
            batch_size, instance_num, seq_len = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            reshape = True
        else:
            reshape = False

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "csqa":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))

        if reshape:
            logits = logits.reshape(batch_size, instance_num)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ElectraForSequenceClassificationCL(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)
        self.cl_linear = nn.Linear(config.hidden_size, config.hidden_size)

        # self.similarity = Similarity(temp=0.05)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        with_cl=False,
    ):
        """
        input shape
        - train [B, 3, L]
        - predict [B, L]
        """
        if with_cl:
            # train [B, 4, L] -> [4B, L]
            batch_size, case_num = input_ids.shape[:2]
            input_ids = input_ids.view((-1, input_ids.shape[-1]))
            token_type_ids = token_type_ids.view((-1, token_type_ids.shape[-1]))
            attention_mask = attention_mask.view((-1, input_ids.shape[-1]))
            labels = labels.view(-1, 1)
        
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        sequence_output = discriminator_hidden_states[0]

        # contrastive loss
        if with_cl:
            # https://pytorch.org/docs/1.2.0/nn.html#tripletmarginloss
            cls_tokens = sequence_output[:,0,:].reshape(batch_size, case_num, self.config.hidden_size)   # cls
            cls_tokens = self.cl_linear(cls_tokens)
            anchor, positive, negative = cls_tokens[:,0,:], cls_tokens[:,1,:], cls_tokens[:,2,:]
            # z1_z2_cos = self.similarity(z1, z2).reshape(-1,1)
            # z1_z3_cos = self.similarity(z1, z3).reshape(-1,1)
            # cos_sim = torch.cat([z1_z2_cos, z1_z3_cos], 1)
            # labels_cl = torch.zeros(batch_size).long().to(cos_sim.device)

            # loss_fct = nn.CrossEntropyLoss()
            # cl_loss = loss_fct(cos_sim, labels_cl)
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            cl_loss = triplet_loss(anchor, positive, negative)

        # clasification loss
        logits = self.classifier(sequence_output)
        cls_loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    cls_loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    cls_loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                cls_loss = loss_fct(logits, labels)

        # import pdb; pdb.set_trace()
        return SequenceClassifierOutput(
            loss=cls_loss + 0.01 * cl_loss if with_cl else cls_loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

# modules for CL

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp