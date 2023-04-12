import argparse
import logging
import os
import math
import optuna
from optuna.trial import TrialState


import numpy as np
import torch
import torch.nn as nn
from tqdm.std import trange
from pprint import pprint
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_metric_learning import losses, miners
from info_nce import InfoNCE

from utils.functions import (
    compute_scores,
    create_dataset_and_dataloader,
    create_model,
    read_config,
    set_seed,
    calculate_cross_entropy_loss,
    calculate_infonce_loss,
    calculate_joint_cross_entropy_and_contarstive_loss,
)

log = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
    model, train_dataset, train_dataloader, loss_fn, optimizer, scheduler
):

    epoch_loss = 0.0

    model.train()

    print_freq = 0

    for step, batch in enumerate(train_dataloader):

        model.zero_grad()

        concepts_batch, property_batch = train_dataset.add_context(batch)

        if print_freq < 1:
            log.info(f"concepts_batch : {concepts_batch}")
            log.info(f"property_batch : {property_batch}")

            print(f"concepts_batch : {concepts_batch}", flush=True)
            print(f"property_batch : {property_batch}", flush=True)

            print_freq += 1

        ids_dict = train_dataset.tokenize(concepts_batch, property_batch)

        if train_dataset.hf_tokenizer_name in ("roberta-base", "roberta-large"):

            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
            ) = [val.to(device) for _, val in ids_dict.items()]

            concept_token_type_id = None
            property_token_type_id = None

        else:
            (
                concept_inp_id,
                concept_attention_mask,
                concept_token_type_id,
                property_input_id,
                property_attention_mask,
                property_token_type_id,
            ) = [val.to(device) for _, val in ids_dict.items()]

        concept_embedding, property_embedding, logits = model(
            concept_input_id=concept_inp_id,
            concept_attention_mask=concept_attention_mask,
            concept_token_type_id=concept_token_type_id,
            property_input_id=property_input_id,
            property_attention_mask=property_attention_mask,
            property_token_type_id=property_token_type_id,
        )

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):

            batch_loss, batch_logits, batch_labels = calculate_cross_entropy_loss(
                dataset=train_dataset,
                batch=batch,
                concept_embedding=concept_embedding,
                property_embedding=property_embedding,
                loss_fn=loss_fn,
                device=device,
            )
        elif isinstance(loss_fn, losses.NTXentLoss):
            pass

            # batch_loss = calculate_ntxent_loss(
            #     dataset=train_dataset,
            #     batch=batch,
            #     concept_embedding=concept_embedding,
            #     property_embedding=property_embedding,
            #     loss_fn=loss_fn,
            #     device=device,
            # )

        elif isinstance(loss_fn, InfoNCE):

            batch_loss = calculate_infonce_loss(
                dataset=train_dataset,
                batch=batch,
                concept_embedding=concept_embedding,
                property_embedding=property_embedding,
                loss_fn=loss_fn,
                device=device,
            )
        elif isinstance(loss_fn, list):

            batch_loss = calculate_joint_cross_entropy_and_contarstive_loss(
                dataset=train_dataset,
                batch=batch,
                concept_embedding=concept_embedding,
                property_embedding=property_embedding,
                loss_fn=loss_fn,
                device=device,
            )

        batch_loss.backward()
        torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        epoch_loss += batch_loss.item()

        if step % 100 == 0 and not step == 0:

            log.info(
                f"Batch {step} of {len(train_dataloader)} ----> Train Batch Loss : {round(batch_loss.item(), 4)}"
            )

    avg_epoch_loss = round(epoch_loss / len(train_dataloader), 4)

    return avg_epoch_loss


def evaluate(model, valid_dataset, valid_dataloader, loss_fn, device):

    val_loss = 0.0

    model.eval()

    epoch_logits, epoch_labels = [], []

    for step, batch in enumerate(valid_dataloader):

        concepts_batch, property_batch = valid_dataset.add_context(batch)

        ids_dict = valid_dataset.tokenize(concepts_batch, property_batch)

        if valid_dataset.hf_tokenizer_name in ("roberta-base", "roberta-large"):

            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
            ) = [val.to(device) for _, val in ids_dict.items()]

            concept_token_type_id = None
            property_token_type_id = None

        else:
            (
                concept_inp_id,
                concept_attention_mask,
                concept_token_type_id,
                property_input_id,
                property_attention_mask,
                property_token_type_id,
            ) = [val.to(device) for _, val in ids_dict.items()]

        with torch.no_grad():

            concept_embedding, property_embedding, logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                concept_token_type_id=concept_token_type_id,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
                property_token_type_id=property_token_type_id,
            )

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):

            batch_loss, batch_logits, batch_labels = calculate_cross_entropy_loss(
                dataset=valid_dataset,
                batch=batch,
                concept_embedding=concept_embedding,
                property_embedding=property_embedding,
                loss_fn=loss_fn,
                device=device,
            )
        elif isinstance(loss_fn, losses.NTXentLoss):
            pass
            # batch_loss = calculate_contrastive_loss(
            #     dataset=valid_dataset,
            #     batch=batch,
            #     concept_embedding=concept_embedding,
            #     property_embedding=property_embedding,
            #     loss_fn=loss_fn,
            #     device=device,
            # )

        elif isinstance(loss_fn, InfoNCE):

            batch_loss = calculate_infonce_loss(
                dataset=valid_dataset,
                batch=batch,
                concept_embedding=concept_embedding,
                property_embedding=property_embedding,
                loss_fn=loss_fn,
                device=device,
            )
        elif isinstance(loss_fn, list):

            batch_loss = calculate_joint_cross_entropy_and_contarstive_loss(
                dataset=valid_dataset,
                batch=batch,
                concept_embedding=concept_embedding,
                property_embedding=property_embedding,
                loss_fn=loss_fn,
                device=device,
            )

        val_loss += batch_loss.item()
        torch.cuda.empty_cache()

    avg_val_loss = round(val_loss / len(valid_dataloader), 2)

    return avg_val_loss


def train(config, trial=None):

    log.info("Initialising datasets...")

    train_dataset, train_dataloader = create_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="train"
    )

    valid_dataset, valid_dataloader = create_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="valid"
    )

    log.info("Initialising Model...")

    model = create_model(config.get("model_params"))
    model.to(device)
    # log.info(f"Model Loaded : {model}")

    # -------------------- Preparation for training  ------------------- #

    lr = config["training_params"]["lr"]
    weight_decay = config["training_params"]["weight_decay"]
    loss_function = config["training_params"]["loss_function"]

    log.info(f"Loss Function Name  : {loss_function}")

    if loss_function == "cross_entropy":
        loss_fn = nn.BCEWithLogitsLoss()

    elif loss_function == "infonce":
        tau = config["training_params"]["tau"]
        loss_fn = InfoNCE(temperature=tau, reduction="mean", negative_mode="unpaired")

    elif loss_function == "joint":

        tau = config["training_params"]["tau"]
        bce_loss = nn.BCEWithLogitsLoss()
        infonce_loss = InfoNCE(
            temperature=tau, reduction="mean", negative_mode="unpaired"
        )

        loss_fn = [bce_loss, infonce_loss]

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,)

    total_training_steps = len(train_dataloader) * config["training_params"].get(
        "max_epochs"
    )

    if config["training_params"]["lr_policy"] == "warmup":

        warmup_ratio = config["training_params"]["warmup_ratio"]
        num_warmup_steps = math.ceil(total_training_steps * warmup_ratio)

    else:
        num_warmup_steps = 0

    log.info(f"Warmup-steps: {num_warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    start_epoch = 1

    epoch_count = []
    train_losses = []
    valid_losses = []

    log.info(f"Training the concept property model on {device}")

    patience_counter = 0

    for epoch in trange(start_epoch, config["training_params"].get("max_epochs") + 1):

        log.info(f"  Epoch {epoch} of {config['training_params'].get('max_epochs')}")
        print("\n", flush=True)

        train_loss = train_single_epoch(
            model=model,
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        log.info(f"Train Epoch {epoch} finished !!")
        log.info(f"  Average Train Loss: {train_loss}")

        # ----------------------------------------------#
        # ----------------------------------------------#
        # ---------------Validation---------------------#
        # ----------------------------------------------#
        # ----------------------------------------------#

        log.info(f"Running Validation ....")
        print(flush=True)

        valid_loss = evaluate(
            model=model,
            valid_dataset=valid_dataset,
            valid_dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_count.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        log.info(f"  Average validation Loss: {valid_loss}")
        print(flush=True)

        if trial is not None:

            print(f"train_losses : {train_losses}", flush=True)
            print("valid_losses : {valid_losses}", flush=True)

            log.info(f"train_losses : {train_losses}")
            log.info(f"valid_losses : {valid_losses}")

            trial.report(valid_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return valid_loss

        else:
            log.info(f"optuna_trial is : {trial}")

            if valid_loss >= best_val_loss:
                patience_counter += 1
                log.info(
                    f"Current validation loss: {valid_loss} is greater than/equal to the previous best loss: {best_val_loss}"
                )
                log.info("Incrementing Patience Counter")
            else:
                patience_counter = 0

                log.info(
                    f"Current epoch {epoch}, validation loss {valid_loss} is less than the previous best loss : {best_val_loss}"
                )
                log.info(f"Resetting the Patience Counter to {patience_counter}")

                # best_val_f1 = val_binary_f1
                best_val_loss = valid_loss

                best_model_path = os.path.join(
                    config["training_params"].get("export_path"),
                    config["model_params"].get("model_name"),
                )

                log.info(f"patience_counter : {patience_counter}")
                log.info(f"best_model_path : {best_model_path}")

                torch.save(
                    model.state_dict(), best_model_path,
                )

                log.info(
                    f"Best model at epoch: {epoch}, Validation Loss : {valid_loss}"
                )
                log.info(f"The model is saved in : {best_model_path}")

            log.info("Validation Scores")
            log.info(f" Best Validation Loss Yet : {best_val_loss}")

            log.info(f"train_losses : {train_losses}")
            log.info(f"valid_losses : {valid_losses}")

            print(f"train_losses : {train_losses}", flush=True)
            print("valid_losses : {valid_losses}", flush=True)

            if patience_counter >= config["training_params"].get(
                "early_stopping_patience"
            ):
                log.info(
                    f"Early Stopping ---> Maximum Patience - {config['training_params'].get('early_stopping_patience')} Reached !!"
                )

                break

    return best_val_loss


def test_best_model(config):

    log.info(f"\n {'*' * 50}")
    log.info(f"Testing the best model")

    model = create_model(config.get("model_params"))

    best_model_path = os.path.join(
        config["training_params"]["export_path"], config["model_params"]["model_name"],
    )

    log.info(f"Testing the best model : {best_model_path}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    test_dataset, test_dataloader = create_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="test"
    )

    label = test_dataset.label
    all_test_preds = []

    for step, batch in enumerate(test_dataloader):

        concepts_batch, property_batch = test_dataset.add_context(batch)

        ids_dict = test_dataset.tokenize(concepts_batch, property_batch)

        (
            concept_inp_id,
            concept_attention_mask,
            concept_token_type_id,
            property_input_id,
            property_attention_mask,
            property_token_type_id,
        ) = [val.to(device) for _, val in ids_dict.items()]

        with torch.no_grad():

            concept_embedding, property_embedding, logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                concept_token_type_id=concept_token_type_id,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
                property_token_type_id=property_token_type_id,
            )

        preds = torch.round(torch.sigmoid(logits))
        all_test_preds.extend(preds.detach().cpu().numpy().flatten())

    test_scores = compute_scores(label, np.asarray(all_test_preds))

    log.info(f"Test Metrices")
    log.info(f"Test labels shape: {label.shape}")
    log.info(f"Test Preds shape: {np.asarray(all_test_preds).shape}")

    for key, value in test_scores.items():
        log.info(f"{key} : {value}")
    print(flush=True)


if __name__ == "__main__":

    set_seed(1)

    parser = argparse.ArgumentParser(description="Biencoder Concept Property Model")

    parser.add_argument(
        "--config_file", required=True, help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The model is run with the following configuration")

    log.info(f"\n {config} \n")
    print(f"Input Config File")
    pprint(config, sort_dicts=False)

    hp_tuning = config["training_params"]["hp_tuning"]

    if not hp_tuning:
        log.info(f"Running the model for choosen parameters ...")
        val_loss = train(config, trial=None)

    # We are not testing the model yet..We will test it on McRae Testset after finetuning
    # test_best_model(config)

    elif hp_tuning == "grid_search":

        log.info("Doing Hyperparameter Search With Grid Search")

        # max_epochs = [15, 20, 25, 30]
        # batch_size = [8, 16, 32, 64]
        # warmup_ratio = [0.1, 0.15]
        # weight_decay = [0.1]

        # tau = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.07]
        # lr = [2e-6]

        # G2 params
        # max_epochs = [50]
        # batch_size = [32, 8, 16, 64]
        # warmup_ratio = [0.6, 0.1, 0.15]
        # weight_decay = [0.1, 0.3]

        # tau = [0.01, 0.05, 0.07, 0.1]
        # lr = [2e-6]
        # hidden_dropout_prob = [0.1, 0.3]

        # G3 Params
        # max_epochs = [4, 6, 8]
        # batch_size = [8, 16, 32]
        # warmup_ratio = [0.1, 0.15]
        # weight_decay = [0.1, 0.3]
        # tau = [0.01, 0.05, 0.07, 0.1]
        # lr = [2e-6, 1e-5]
        # hidden_dropout_prob = [0.1]

        # G4 Large batch size grid
        max_epochs = [10, 15, 18, 20]
        batch_size = [16, 32, 64, 128]
        warmup_ratio = [0.15]
        weight_decay = [0.9]
        tau = [0.9]
        lr = [1e-5]
        hidden_dropout_prob = [0.4]

        log.info(f"max_epochs : {max_epochs}")
        log.info(f"batch_size : {batch_size}")
        log.info(f"warmup_ratio : {warmup_ratio}")
        log.info(f"weight_decay : {weight_decay}")
        log.info(f"tau : {tau}")
        log.info(f"lr : {lr}")
        log.info(f"hidden_dropout_prob : {hidden_dropout_prob}")

        hf_checkpoint_name = config["model_params"]["hf_checkpoint_name"]
        model_prefix = config["model_params"]["model_name"]

        for me in max_epochs:
            for bs in batch_size:
                for wr in warmup_ratio:
                    for wd in weight_decay:
                        for t in tau:
                            for l in lr:
                                for do in hidden_dropout_prob:

                                    discription_str = f"ep{me}_bs{bs}_wr{wr}_wd{wd}_tau{t}_lr{l}_do{do}"

                                    config["training_params"]["max_epochs"] = me
                                    config["dataset_params"]["loader_params"][
                                        "batch_size"
                                    ] = bs
                                    config["training_params"]["warmup_ratio"] = wr
                                    config["training_params"]["weight_decay"] = wd

                                    config["training_params"]["tau"] = t
                                    config["training_params"]["lr"] = l
                                    config["model_params"]["hidden_dropout_prob"] = do

                                    config["model_params"]["model_name"] = (
                                        model_prefix
                                        + "_"
                                        + hf_checkpoint_name.replace("-", "_")
                                        + "_"
                                        + discription_str
                                        + ".pt"
                                    )

                                    log.info("\n")
                                    log.info("*" * 50)

                                    log.info(f"discription_str : {discription_str}")

                                    log.info(
                                        f"new_model_run : max_epochs: {me}, batch_size: {bs}, warmup_ratio : {wr}, weight_decay : {wd}, tau: {t}, lr: {lr}, dropout: {do}"
                                    )
                                    log.info(
                                        f"model_name: {config['model_params']['model_name']}"
                                    )
                                    log.info(f"new_config_file")
                                    log.info(config)

                                    val_loss = train(config, trial=None)

    elif hp_tuning == "optuna":

        log.info("Doing Hyperparameter Search with Optuna")

        def objective(trial):

            _batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            _max_epochs = trial.suggest_int("max_epochs", 10, 60)
            _warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.20)

            _tau = trial.suggest_float("tau", 0.01, 0.06)
            _lr = trial.suggest_float("lr", 2e-6, 5e-5, log=True)

            config["dataset_params"]["loader_params"]["batch_size"] = _batch_size
            config["training_params"]["max_epochs"] = _max_epochs
            config["training_params"]["warmup_ratio"] = _warmup_ratio

            config["training_params"]["tau"] = _tau
            config["training_params"]["lr"] = _lr

            log.info("Optuna Selected Params")

            log.info(f"batch_size : {_batch_size}")
            log.info(f"max_epochs : {_max_epochs}")
            log.info(f"warmup_ratio : {_warmup_ratio}")

            log.info(f"tau : {_tau}")
            log.info(f"lr : {_lr}")

            hf_model_id = config["model_params"]["hf_checkpoint_name"].replace("-", "_")

            model_name = f"biencoder_cnetp_{hf_model_id}_bs{_batch_size}_ep{_max_epochs}_wr{_warmup_ratio}_tau{_tau}_lr{_lr}.pt"

            config["model_params"]["model_name"] = model_name

            log.info(f"Runnig with Config File")
            log.info(config)

            val_loss = train(config=config, trial=trial)

            return val_loss

        def hp_tune(objective):

            optuna_sampler = optuna.samplers.TPESampler()
            optuna_pruner = optuna.pruners.SuccessiveHalvingPruner()

            study = optuna.create_study(
                study_name="Contrastive BiEncoder",
                direction="minimize",
                sampler=optuna_sampler,
                pruner=optuna_pruner,
            )

            study.optimize(objective, n_trials=100)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(
                deepcopy=False, states=[TrialState.COMPLETE]
            )

            print("Study statistics: ", flush=True)
            print("  Number of finished trials: ", len(study.trials), flush=True)
            print("  Number of pruned trials: ", len(pruned_trials), flush=True)
            print("  Number of complete trials: ", len(complete_trials), flush=True)

            print("Best trial:", flush=True)
            trial = study.best_trial

            print("  Value: ", trial.value, flush=True)

            print("  Params: ", flush=True)
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value), flush=True)

        hp_tune(objective=objective)

