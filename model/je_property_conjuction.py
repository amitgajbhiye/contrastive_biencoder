import os
import sys
import time
import logging


sys.path.insert(0, os.getcwd())

import warnings
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_convert
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils.je_utils import compute_scores, read_config, set_seed

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_logger(config):

    log_file_name = os.path.join(
        "logs",
        config.get("log_dirctory"),
        f"log_{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
    )

    print("config.get('experiment_name') :", config.get("experiment_name"), flush=True)
    print("\nlog_file_name :", log_file_name, flush=True)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(name)s : %(levelname)s - %(message)s",
    )


context_templates = {
    1: [
        "concept <con> can be described as <prop_list>.",
        "concept <con> can be described as <predict_prop>.",
    ],
    2: [
        "concept <con> can be described as <prop_list>?",
        "<[MASK]>, concept <con> can be described as <predict_prop>.",
    ],
}


class DatasetPropConjuction(Dataset):
    def __init__(self, concept_property_file, dataset_params):

        if isinstance(concept_property_file, pd.DataFrame):

            self.data_df = concept_property_file
            log.info(
                f"Supplied Concept Property File is a Dataframe : {self.data_df.shape}",
            )

        elif os.path.isfile(concept_property_file):

            log.info(
                f"Supplied Concept Property File is a Path : {concept_property_file}"
            )
            log.info(f"Loading into Dataframe ... ")

            self.data_df = pd.read_csv(
                concept_property_file,
                sep="\t",
                header=None,
                names=["concept", "conjuct_prop", "predict_prop", "labels"],
                dtype={
                    "concept": str,
                    "conjuct_prop": str,
                    "predict_prop": str,
                    "labels": int,
                },
            )[0:1000]

            log.info(f"Loaded Dataframe Shape: {self.data_df.shape}")

        else:
            raise TypeError(
                f"Input file type is not correct !!! - {concept_property_file}"
            )

        self.data_df.reset_index(inplace=True, drop=True)

        self.hf_tokenizer_name = dataset_params["hf_tokenizer_name"]
        self.hf_tokenizer_path = dataset_params["hf_tokenizer_path"]

        self.tokenizer = BertTokenizer.from_pretrained(self.hf_tokenizer_path)
        self.max_len = dataset_params["max_len"]

        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        self.context_id = dataset_params["context_id"]

        log.info(f"Context ID : {self.context_id}")

        if self.context_id:
            log.info(f"Adding Context : {context_templates[self.context_id]}")

        self.print_freq = 0

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        concept = self.data_df["concept"][idx].replace(".", "").strip()
        conjuct_props = self.data_df["conjuct_prop"][idx].strip()
        predict_prop = self.data_df["predict_prop"][idx].replace(".", "").strip()
        labels = torch.tensor(self.data_df["labels"][idx])

        # print(f"Data Row : {self.data_df[idx].to_list()}", flush=True)

        if conjuct_props == "no_similar_property":
            conjuct_props = ""
        else:

            conjuct_props = conjuct_props.split(", ")

            if len(conjuct_props) >= 2:

                conjuct_props[-1] = "and " + conjuct_props[-1]
                conjuct_props = ", ".join(conjuct_props)
            else:

                conjuct_props = ", ".join(conjuct_props)

        if self.context_id == 1:

            # NLI Formulation
            # sent_1 = "concept <con> can be described as <prop_list>.
            # sent_2 = "concept <con> can be described as <predict_prop>.

            con_prop_template, predict_prop_template = context_templates[
                self.context_id
            ]

            sent_1 = con_prop_template.replace("<con>", concept).replace(
                "<prop_list>", conjuct_props
            )

            sent_2 = predict_prop_template.replace("<con>", concept).replace(
                "<predict_prop>", predict_prop
            )

        elif self.context_id == 2:

            # MLM Formulation
            # sent_1 = concept <con> can be described as <prop_list> ?
            # sent_2 = <[MASK]>, concept <con> can be described as <predict_prop>.

            # 2: ["concept <con> can be described as <prop_list> ?",
            # "<[MASK]>, concept <con> can be described as <predict_prop>.",]

            con_prop_template, predict_prop_template = context_templates[
                self.context_id
            ]

            sent_1 = con_prop_template.replace("<con>", concept).replace(
                "<prop_list>", conjuct_props
            )

            sent_2 = (
                predict_prop_template.replace("<[MASK]>", self.mask_token)
                .replace("<con>", concept)
                .replace("<predict_prop>", predict_prop)
            )

        # print(f"sent_1 : {sent_1}", flush=True)
        # print(f"sent_2 : {sent_2}", flush=True)
        # print(flush=True)

        # ++++++++++++++++++++++++

        # if conjuct_props == "no_similar_property":

        #     con_prop_conj = concept + " " + self.sep_token
        #     prop_to_predict = predict_prop

        # else:

        #     con_prop_conj = concept + " " + self.sep_token + " " + conjuct_props
        #     prop_to_predict = predict_prop

        encoded_dict = self.tokenizer.encode_plus(
            text=sent_1,
            text_pair=sent_2,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        input_ids = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]
        token_type_ids = encoded_dict["token_type_ids"]

        if self.print_freq < 2:

            print(flush=True)
            print(f"sent_1 : {sent_1}", flush=True)
            print(f"sent_2 : {sent_2}", flush=True)
            print(
                f"Decoded Sent - {self.tokenizer.decode(input_ids.squeeze())}",
                flush=True,
            )

            self.print_freq += 1
            print(flush=True)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ModelPropConjuctionJoint(nn.Module):
    def __init__(self, model_params):
        super(ModelPropConjuctionJoint, self).__init__()

        self.hf_checkpoint_name = model_params["hf_checkpoint_name"]
        self.hf_model_path = model_params["hf_model_path"]
        self.num_labels = model_params["num_labels"]
        self.context_id = model_params["context_id"]

        self.bert = BertForSequenceClassification.from_pretrained(
            self.hf_model_path, num_labels=self.num_labels, output_hidden_states=True
        )

        assert self.bert.config.num_labels == 2

        classifier_dropout = self.bert.config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        loss_fct = nn.BCEWithLogitsLoss()

        if self.context_id == 1:

            output = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss, logits = output.loss, output.logits

            return loss, logits

        elif self.context_id == 2:

            output = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            hidden_states = output.hidden_states

            def get_mask_token_embeddings(last_layer_hidden_states):

                BERT_MASK_TOKEN_ID = 103

                _, mask_token_index = (
                    input_ids == torch.tensor(BERT_MASK_TOKEN_ID)
                ).nonzero(as_tuple=True)

                mask_vectors = torch.vstack(
                    [
                        torch.index_select(v, 0, torch.tensor(idx))
                        for v, idx in zip(last_layer_hidden_states, mask_token_index)
                    ]
                )

                return mask_vectors

            mask_vectors = get_mask_token_embeddings(
                last_layer_hidden_states=hidden_states[-1]
            )

            mask_vectors = self.dropout(mask_vectors)
            mask_logits = self.classifier(mask_vectors).view(-1)
            labels = labels.view(-1).float()

            print("self.context_id : {self.context_id}", flush=True)
            print(f"Mask Vector Shape : {mask_vectors.shape}", flush=True)
            print(f"Mask Logit Shape : {mask_logits.shape}", flush=True)
            print(f"Labels Shape :{labels.shape}", flush=True)

            print(f"mask_logits : {mask_logits}", flush=True)
            print(f"labels : {labels}", flush=True)

            mask_loss = loss_fct(mask_logits, labels)

            print(f"Mask Loss : {mask_loss}", flush=True)

            return mask_loss, mask_logits


def prepare_data_and_models(
    config, train_file, valid_file=None, test_file=None,
):

    training_params = config["training_params"]

    load_pretrained = training_params["load_pretrained"]
    pretrained_model_path = training_params["pretrained_model_path"]
    lr = training_params["lr"]
    max_epochs = training_params["max_epochs"]
    num_warmup_steps = training_params["num_warmup_steps"]
    batch_size = training_params["batch_size"]

    dataset_params = config["dataset_params"]
    model_params = config["model_params"]

    num_workers = 4

    train_data = DatasetPropConjuction(train_file, dataset_params)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=default_convert,
        num_workers=num_workers,
        pin_memory=True,
    )

    log.info(f"Train Data DF shape : {train_data.data_df.shape}")

    if valid_file is not None:
        val_data = DatasetPropConjuction(valid_file, dataset_params)
        val_sampler = RandomSampler(val_data)
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=default_convert,
            num_workers=num_workers,
            pin_memory=True,
        )
        log.info(f"Valid Data DF shape : {val_data.data_df.shape}")
    else:
        log.info(f"Validation File is Empty.")
        val_dataloader = None

    if test_file is not None:
        test_data = DatasetPropConjuction(test_file, dataset_params)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=default_convert,
            num_workers=num_workers,
            pin_memory=True,
        )
        log.info(f"Test Data DF shape : {test_data.data_df.shape}")
    else:
        log.info("Test File is Empty.")
        test_dataloader = None

    log.info(f"Load Pretrained : {load_pretrained}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")

    if load_pretrained:

        log.info(f"Loading Pretrained Model From : {pretrained_model_path}")

        model = ModelPropConjuctionJoint(model_params)
        model.load_state_dict(torch.load(pretrained_model_path))

        log.info(f"Loaded Pretrained Model")

    else:

        log.info(f"Training the Model from Scratch ...")
        model = ModelPropConjuctionJoint(model_params)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    return (
        model,
        scheduler,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


def train_on_single_epoch(model, scheduler, optimizer, train_dataloader):

    train_losses = []

    model.train()
    for step, batch in enumerate(train_dataloader):

        model.zero_grad()

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
            device
        )
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        loss, logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        train_losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 and not step == 0:
            log.info(
                "   Batch {} of Batch {} ---> Batch Loss {}".format(
                    step, len(train_dataloader), round(loss.item(), 4)
                )
            )

    avg_train_loss = round(np.mean(train_losses), 4)

    log.info("Average Train Loss :", avg_train_loss)

    return avg_train_loss, model


def evaluate(model, dataloader):

    model.eval()

    val_losses, val_accuracy, val_preds, val_labels = [], [], [], []

    for step, batch in enumerate(dataloader):

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
            device
        )
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        val_losses.append(loss.item())

        if model.context_id == 1:
            batch_preds = torch.argmax(logits, dim=1).flatten()
        elif model.context_id == 2:
            batch_preds = torch.round(torch.sigmoid(logits))

        val_preds.extend(batch_preds.cpu().detach().numpy())
        val_labels.extend(labels.cpu().detach().numpy())

    avg_val_loss = round(np.mean(val_losses), 4)

    val_preds = np.array(val_preds).flatten()
    val_labels = np.array(val_labels).flatten()

    return avg_val_loss, val_preds, val_labels


def train(
    training_params,
    model,
    scheduler,
    optimizer,
    train_dataloader,
    val_dataloader=None,
    test_dataloader=None,
    fold=None,
):

    max_epochs = training_params["max_epochs"]
    model_name = training_params["model_name"]
    save_dir = training_params["save_dir"]

    if fold is not None:
        best_model_path = os.path.join(save_dir, f"{fold}_{model_name}")
    else:
        best_model_path = os.path.join(save_dir, model_name)

    patience_early_stopping = training_params["patience_early_stopping"]

    best_valid_f1 = 0.0
    patience_counter = 0
    start_epoch = 1
    train_losses, valid_losses = [], []

    for epoch in range(start_epoch, max_epochs + 1):

        log.info("Epoch {:} of {:}".format(epoch, max_epochs))

        train_loss, model = train_on_single_epoch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
        )

        if val_dataloader is not None:

            log.info(f"Running Validation ....")

            valid_loss, valid_preds, valid_gold_labels = evaluate(
                model=model, dataloader=val_dataloader
            )

            scores = compute_scores(valid_gold_labels, valid_preds)
            valid_binary_f1 = scores["binary_f1"]

            if best_valid_f1 > valid_binary_f1:
                patience_counter += 1
            else:
                patience_counter = 0
                best_valid_f1 = valid_binary_f1

                log.info("\n '+' * 20")
                log.info("Saving Best Model at Epoch :", epoch, model_name)
                log.info("Epoch :", epoch)
                log.info("   Best Validation F1:", best_valid_f1)

                torch.save(model.state_dict(), best_model_path)

                log.info(f"The best model is saved at : {best_model_path}")

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            log.info("+" * 50)

            log.info(f"valid_preds shape: {valid_preds.shape}")
            log.info(f"val_gold_labels shape: {valid_gold_labels.shape}")

            log.info(f"Current Validation F1 Score Binary {valid_binary_f1}")
            log.info(f"Best Validation F1 Score Yet : {best_valid_f1}")

            log.info(f"Training Loss: {train_loss}")
            log.info(f"Validation Loss: {valid_loss}")

            log.info("Validation Scores")
            for key, value in scores.items():
                log.info(f" {key} :  {value}")

            if patience_counter >= patience_early_stopping:

                log.info(f"Train Losses :", train_losses)
                log.info(f"Validation Losses: ", valid_losses)

                log.info("Early Stopping ---> Patience Reached!!!")
                break

    log.info(
        f"Saving the model after training - {max_epochs} Epochs on Fold Number: {fold}!!!"
    )
    torch.save(model.state_dict(), best_model_path)
    log.info(f"The model is save at : {best_model_path}")

    if test_dataloader is not None:
        log.info(f"Testing the Model on Fold : {fold}")

        _, test_preds, test_gold_labels = evaluate(
            model=model, dataloader=test_dataloader
        )

        scores = compute_scores(test_gold_labels, test_preds)

        log.info(f"Fold: {fold}, test_gold_labels.shape : {test_gold_labels.shape}")
        log.info(f"Fold: {fold}, test_preds.shape : {test_preds.shape}")

        assert (
            test_gold_labels.shape == test_preds.shape
        ), "shape of fold's labels not equal to fold's preds"

        log.info(f"************ Test Scores Fold {fold} ************")
        for key, value in scores.items():
            log.info(f" {key} :  {value}")

        return test_preds, test_gold_labels


################ Finetuning Code Starts Here ################


def do_cv(config):

    training_params = config["training_params"]

    cv_type = training_params["cv_type"]
    data_dir = training_params["data_dir"]
    save_prefix = training_params["save_prefix"]

    log.info(f"CV Type : {cv_type}")

    if cv_type == "concept_split":

        train_file_base_name = "train_mcrae"
        test_file_base_name = "test_mcrae"

        train_file_name = os.path.join(
            data_dir, f"{save_prefix}_{train_file_base_name}.tsv"
        )
        test_file_name = os.path.join(
            data_dir, f"{save_prefix}_{test_file_base_name}.tsv"
        )

        log.info(f"Train File Name : {train_file_name}")
        log.info(f"Test File Name : {test_file_name}")

        (
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = prepare_data_and_models(
            config=config,
            train_file=train_file_name,
            valid_file=None,
            test_file=test_file_name,
        )

        test_preds, test_gold_labels = train(
            training_params,
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            fold=None,
        )

    elif cv_type in ("property_split", "concept_property_split"):

        if cv_type == "property_split":

            num_fold = 5
            train_file_base_name = "train_prop_conj"
            test_file_base_name = "test_prop_conj"

        elif cv_type == "concept_property_split":

            num_fold = 9
            train_file_base_name = "--------------"
            test_file_base_name = "--------------"

        else:
            raise NameError(
                "Specify cv_type from : 'concept_split', 'property_split', 'concept_property_split'"
            )

        log.info(f"Number of Folds : {num_fold}")
        log.info(f"Data Dir : {data_dir}")
        log.info(f"Train File Base Name : {train_file_base_name}")
        log.info(f"Test File Base Name : {test_file_base_name}")

        all_folds_test_preds, all_folds_test_labels = [], []
        for fold in range(num_fold):

            log.info("*" * 50)
            log.info(f"Training the model on Fold : {fold} of {num_fold}")
            log.info("*" * 50)

            train_file_name = os.path.join(
                data_dir, f"{save_prefix}_{fold}_{train_file_base_name}_{cv_type}.tsv"
            )
            test_file_name = os.path.join(
                data_dir, f"{save_prefix}_{fold}_{test_file_base_name}_{cv_type}.tsv"
            )

            log.info(f"Train File Name : {train_file_name}")
            log.info(f"Test File Name : {test_file_name}")

            (
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = prepare_data_and_models(
                config=config,
                train_file=train_file_name,
                valid_file=None,
                test_file=test_file_name,
            )

            fold_test_preds, fold_test_gold_labels = train(
                training_params,
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                fold=fold,
            )

            all_folds_test_preds.extend(fold_test_preds)
            all_folds_test_labels.extend(fold_test_gold_labels)

        all_folds_test_preds = np.array(all_folds_test_preds).flatten()
        all_folds_test_labels = np.array(all_folds_test_labels).flatten()

        log.info(f"Shape of All Folds Preds : {all_folds_test_preds.shape}")
        log.info(f"Shape of All Folds Labels : {all_folds_test_labels.shape}")

        assert (
            all_folds_test_preds.shape == all_folds_test_labels.shape
        ), "shape of all folds labels not equal to all folds preds"

        log.info("*" * 50)
        log.info(f"Calculating the scores for All Folds")
        print(f"Calculating the scores for All Folds", flush=True)

        scores = compute_scores(all_folds_test_labels, all_folds_test_preds)

        for key, value in scores.items():
            log.info(f"{key} : {value}")
            print(f"{key} : {value}", flush=True)


if __name__ == "__main__":

    set_seed(131)

    parser = ArgumentParser(description="Joint Encoder Property Augmentation Model")

    parser.add_argument(
        "-c", "--config_file", required=True, help="path to the configuration file",
    )

    args = parser.parse_args()
    config = read_config(args.config_file)

    set_logger(config=config)
    log = logging.getLogger(__name__)

    log.info("The model is run with the following configuration")
    log.info(f"\n {config} \n")
    pprint(config, sort_dicts=False)

    training_params = config["training_params"]  # Training Parameters

    pretrain = training_params["pretrain"]
    finetune = training_params["finetune"]

    log.info(f"Pretrain : {pretrain}")
    log.info(f"Finetune : {finetune}")

    if pretrain:

        train_file = training_params["train_file_path"]
        valid_file = training_params["val_file_path"]

        log.info(f"Train File  : {train_file}")
        log.info(f"Valid File  : {valid_file}")

        (
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = prepare_data_and_models(
            config=config, train_file=train_file, valid_file=valid_file, test_file=None,
        )

        train(
            training_params=training_params,
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            fold=None,
        )

    elif finetune:

        hp_tuning = training_params["hp_tuning"]

        if not hp_tuning:

            do_cv(config=config)

        else:

            log.info("Grid Search - Hyperparameter Tuning")

            epochs = [8, 10, 12, 16, 20]
            batch_size = [8, 16, 32, 64]
            learning_rate = [1e-5, 2e-5, 5e-5, 1e-6, 2e-6, 5e-5]

            log.info(f"Max Epochs :  {epochs}")
            log.info(f"Batch Sizes : {batch_size}")
            log.info(f"Learning Rates : {learning_rate}")

            for ep in epochs:
                for bs in batch_size:
                    for lr in learning_rate:

                        log.info("*" * 60)
                        log.info(f"New Run : Epoch: {ep}, Batch Size: {bs}, LR: {lr}")
                        log.info("*" * 60)

                        config["training_params"]["max_epochs"] = ep
                        config["training_params"]["batch_size"] = bs
                        config["training_params"]["lr"] = lr

                        print(
                            f"Running With Params Max Epochs, Batch Size, LR :",
                            config["training_params"]["max_epochs"],
                            config["training_params"]["batch_size"],
                            config["training_params"]["lr"],
                        )

                        print(flush=True)
                        print(f"Running with new config", flush=True)
                        pprint(config, sort_dicts=False)

                        do_cv(config=config)

                        print(flush=True)

