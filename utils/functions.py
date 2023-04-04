import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from data.concept_property_dataset import ConceptPropertyDataset, TestDataset
from data.mcrae_dataset import McRaeConceptPropertyDataset
from model.concept_property_model import ConceptPropertyModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import matplotlib.pyplot as plt


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_logger(config):

    # log_file_name = f"logs/cslb_fine_tuned_100k_logs/log_{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt"

    log_file_name = os.path.join(
        "logs",
        config.get("log_dirctory"),
        f"log_{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
    )

    print("config.get('experiment_name') :", config.get("experiment_name"))
    print("\nlog_file_name :", log_file_name)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(levelname)s : %(name)s - %(message)s",
    )


log = logging.getLogger(__name__)


def to_cpu(x):
    return x.cpu().numpy()


def read_config(config_file):

    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            config_dict = json.load(json_file)
            set_logger(config_dict)
            return config_dict
    else:
        return config_file


def read_train_data(dataset_params):

    data_df = pd.read_csv(
        dataset_params.get("train_file_path"),
        sep="\t",
        header=None,
        names=["concept", "property", "label"],
    )

    data_df.drop_duplicates(inplace=True)
    data_df.dropna(inplace=True)

    log.info(f"Concept Column is null ? : {data_df['concept'].isnull().any()}")
    log.info(f"Property Column is null ? : {data_df['property'].isnull().any()}")
    log.info(f"Label Column is null ? : {data_df['label'].isnull().any()}")

    data_df.dropna(subset=["concept"], inplace=True)
    data_df.dropna(subset=["property"], inplace=True)
    data_df.dropna(subset=["label"], inplace=True)

    data_df = data_df.sample(frac=1)
    data_df.reset_index(drop=True, inplace=True)

    log.info(f"Total Data size {data_df.shape}")

    return data_df[["concept", "property"]], data_df[["label"]]


def read_train_and_test_data(dataset_params):

    train_df = pd.read_csv(
        dataset_params.get("train_file_path"),
        sep="\t",
        header=None,
        names=["concept", "property", "label"],
    )

    test_df = pd.read_csv(
        dataset_params.get("test_file_path"),
        sep="\t",
        header=None,
        names=["concept", "property", "label"],
    )

    train_and_test_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)

    log.info(
        f"Concept Column is null ? : {train_and_test_df['concept'].isnull().any()}"
    )
    log.info(
        f"Property Column is null ? : {train_and_test_df['property'].isnull().any()}"
    )
    log.info(f"Label Column is null ? : {train_and_test_df['label'].isnull().any()}")

    train_and_test_df.dropna(subset=["concept"], inplace=True)
    train_and_test_df.dropna(subset=["property"], inplace=True)
    train_and_test_df.dropna(subset=["label"], inplace=True)
    train_and_test_df.dropna(how="any", inplace=True)

    train_and_test_df.drop_duplicates(inplace=True)

    train_and_test_df = train_and_test_df.sample(frac=1)

    train_and_test_df.reset_index(inplace=True, drop=True)

    print()
    print("+++++++++++ Index Unique +++++++++++++")
    print(train_and_test_df.index.is_unique)
    print(
        f"Duplicated Index Unique : {train_and_test_df.loc[train_and_test_df.index.duplicated(), :]}"
    )
    print("train_and_test_df")
    print(train_and_test_df.head())
    print()

    train_and_test_df["con_id"] = int(-1)
    train_and_test_df["prop_id"] = int(-2)

    unique_concept = train_and_test_df["concept"].unique()
    unique_property = train_and_test_df["property"].unique()

    num_unique_concept = len(unique_concept)
    num_unique_property = len(unique_property)

    log.info(f"Train and Test Data DF shape : {train_and_test_df.shape}")

    log.info(
        f"Number of Unique Concepts in Train and Test Combined DF : {num_unique_concept}"
    )
    log.info(f"Unique Concepts in Train and Test Combined DF : {unique_concept}")

    log.info(
        f"Number of Unique Property in Train and Test Combined DF : {num_unique_property}"
    )
    log.info(f"Unique Property in Train and Test Combined DF : {unique_property}")

    con_to_id_dict = {con: id for id, con in enumerate(unique_concept)}
    con_ids = list(con_to_id_dict.values())
    log.info(f"Concept ids in con_to_id_dict : {con_ids}")

    train_and_test_df.set_index("concept", inplace=True)

    for con in unique_concept:
        train_and_test_df.loc[con, "con_id"] = con_to_id_dict.get(con)

    train_and_test_df.reset_index(inplace=True)

    log.info("Train Test DF after assigning 'con_id'")
    log.info(train_and_test_df.sample(n=10))

    assert sorted(con_ids) == sorted(
        train_and_test_df["con_id"].unique()
    ), "Assigned 'con_ids' do not match"

    #################################
    prop_to_id_dict = {prop: id for id, prop in enumerate(unique_property)}
    prop_ids = list(prop_to_id_dict.values())

    log.info(f"Number of Property ids in prop_to_id_dict : {len(prop_ids)}")
    log.info(f"Property ids in prop_to_id_dict : {prop_ids}")

    train_and_test_df.set_index("property", inplace=True)

    for prop in unique_property:
        train_and_test_df.loc[prop, "prop_id"] = prop_to_id_dict.get(prop)

    train_and_test_df.reset_index(inplace=True)

    log.info("Train Test DF after assigning 'prop_id'")
    log.info(train_and_test_df.sample(n=15))

    assert sorted(prop_ids) == sorted(
        train_and_test_df["prop_id"].unique()
    ), "Assigned 'prop_ids' do not match"

    train_and_test_df.to_csv("cslb_train_and_test.tsv", sep="\t")

    return train_and_test_df


def create_dataset_and_dataloader(dataset_params, dataset_type):

    if dataset_type in ("train", "valid"):
        dataset = ConceptPropertyDataset(dataset_params, dataset_type)
        data_sampler = RandomSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_params["loader_params"]["batch_size"],
            sampler=data_sampler,
            num_workers=dataset_params["loader_params"]["num_workers"],
            pin_memory=dataset_params["loader_params"]["pin_memory"],
        )

    elif dataset_type == "test":

        dataset = TestDataset(dataset_params)
        data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_params["loader_params"]["batch_size"],
            sampler=data_sampler,
            num_workers=dataset_params["loader_params"]["num_workers"],
            pin_memory=dataset_params["loader_params"]["pin_memory"],
        )

    return dataset, dataloader


def create_model(model_params):
    log.info(f"Creating a model from scratch")
    return ConceptPropertyModel(model_params)


def compute_scores(labels, preds):

    assert len(labels) == len(
        preds
    ), f"labels len: {len(labels)} is not equal to preds len {len(preds)}"

    scores = {
        "binary_f1": round(f1_score(labels, preds, average="binary"), 4),
        "micro_f1": round(f1_score(labels, preds, average="micro"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "classification report": classification_report(labels, preds, labels=[0, 1]),
        "confusion matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }

    return scores


# def calculate_contrastive_loss(
#     dataset, batch, concept_embedding, property_embedding, loss_fn, device
# ):
#     embed_all = torch.cat([concept_embedding, property_embedding], dim=0).to(device)

#     # log.info(f"In calculate_contrastive_loss function")
#     # log.info(f"concept_embedding.shape : {concept_embedding.shape}")
#     # log.info(f"property_embedding.shape : {property_embedding.shape}")
#     # log.info(f"embed_all.shape : {embed_all.shape}")

#     indices = torch.arange(0, concept_embedding.size(0), device=device)
#     labels = torch.cat((indices, indices), dim=0)

#     # log.info(f"Labels : {labels}")

#     loss = loss_fn(embed_all, labels)

#     # log.info(f"Loss value in calculate_contrastive_loss function:  {loss.item()}")

#     return loss


def calculate_infonce_loss(
    dataset, batch, concept_embedding, property_embedding, loss_fn, device
):
    query_embedding = dataset.query_embedding

    if query_embedding == "concept":

        loss = loss_fn(
            query=concept_embedding,
            positive_key=property_embedding,
            negative_keys=property_embedding,
        )
    elif query_embedding == "property":

        loss = loss_fn(
            query=property_embedding,
            positive_key=concept_embedding,
            negative_keys=concept_embedding,
        )

    return loss


# def calculate_infonce_loss(
#     dataset, batch, concept_embedding, property_embedding, loss_fn, device
# ):

#     concept_id_list_for_batch = torch.tensor(
#         [dataset.concept2idx[concept] for concept in batch[0]], device=device
#     )
#     property_id_list_for_batch = torch.tensor(
#         [dataset.property2idx[prop] for prop in batch[1]], device=device
#     )

#     # print(flush=True)
#     # print("concept_id_list_for_batch :", concept_id_list_for_batch)
#     # print("property_id_list_for_batch :", property_id_list_for_batch)

#     loss = 0.0

#     for i in range(len(concept_id_list_for_batch)):

#         concept_id = concept_id_list_for_batch[i]
#         property_id = property_id_list_for_batch[i]

#         # print(flush=True)
#         # print(f"Processing:", flush=True)
#         # print(f"concept_id : {concept_id}", flush=True)
#         # print(f"property_id : {property_id}", flush=True)

#         # Extracting the property of the concept at the whole dataset level.
#         property_id_list_for_concept = torch.tensor(
#             dataset.con_pro_dict[concept_id.item()], device=device
#         )

#         # print(
#         #     f"property_id_list_for_concept : {property_id_list_for_concept}", flush=True
#         # )

#         # Extracting the negative property by excluding the properties that the concept may have at the  whole dataset level
#         negative_property_id_for_concept = torch.tensor(
#             [
#                 x
#                 for x in property_id_list_for_batch
#                 if x not in property_id_list_for_concept
#             ],
#             device=device,
#         )

#         # print(
#         #     f"negative_property_id_for_concept : {negative_property_id_for_concept}",
#         #     flush=True,
#         # )

#         # Adding the positive property to the concept so that in the denominator of the loss function calculation
#         # it contains the positive property.
#         negative_property_id_for_concept = torch.cat(
#             (torch.tensor(property_id).unsqueeze(0), negative_property_id_for_concept),
#             dim=0,
#         )

#         print(
#             f"After adding positive property id negative_property_id_for_concept : {negative_property_id_for_concept},\
#                 {negative_property_id_for_concept.shape}",
#             flush=True,
#         )

#         mask_positive_property_for_concept = torch.tensor(
#             [
#                 [1] if x in negative_property_id_for_concept else [0]
#                 for x in property_id_list_for_batch
#             ],
#             device=device,
#         )

#         # print(
#         #     f"mask_positive_property_for_concept : {mask_positive_property_for_concept}",
#         #     flush=True,
#         # )

#         neg_property_embedding = torch.mul(
#             property_embedding, mask_positive_property_for_concept
#         )

#         # print(
#         #     f"concept_embedding[i].shape : {concept_embedding[i].shape}", flush=True,
#         # )

#         # print(
#         #     f"property_embedding[i].shape : {property_embedding[i].shape}", flush=True,
#         # )

#         # print(
#         #     f"neg_property_embedding : {neg_property_embedding.shape}", flush=True,
#         # )

#         loss_for_concept = (
#             loss_fn(
#                 query=concept_embedding[i].unsqueeze(0),
#                 positive_key=property_embedding[i].unsqueeze(0),
#                 negative_keys=neg_property_embedding,
#             )
#             / concept_embedding.shape[0]
#         )

#         print(f"loss_for_concept 1: {loss_for_concept}", flush=True)

#         loss_for_concept = (
#             loss_fn(
#                 query=concept_embedding[i].unsqueeze(0),
#                 positive_key=property_embedding[i].unsqueeze(0),
#                 negative_keys=neg_property_embedding,
#             )
#             / negative_property_id_for_concept.shape[0]
#         )
#         print(f"loss_for_concept 2: {loss_for_concept}", flush=True)

#         loss += loss_for_concept

#         # print(f"Loss for concept : {loss_for_concept}", flush=True)
#         # print(f"Total Loss After adding Concept Loss : {loss}", flush=True)

#     loss_1 = loss_fn(
#         query=concept_embedding,
#         positive_key=property_embedding,
#         negative_keys=property_embedding,
#     )

#     print("*" * 50)
#     print(f"loss_con_by_con : {loss}", flush=True)
#     print(f"total_loss_1: {loss_1}", flush=True)
#     print("*" * 50)

#     return loss


# def calculate_ntxent_loss(
#     dataset, batch, concept_embedding, property_embedding, loss_fn, device
# ):

#     concept_id_list_for_batch = torch.tensor(
#         [dataset.concept2idx[concept] for concept in batch[0]], device=device
#     )
#     property_id_list_for_batch = torch.tensor(
#         [dataset.property2idx[prop] for prop in batch[1]], device=device
#     )

#     print(flush=True)
#     print(
#         "concept_id_list_for_batch :",
#         len(concept_id_list_for_batch),
#         concept_id_list_for_batch,
#         flush=True,
#     )
#     print(
#         "property_id_list_for_batch :",
#         len(property_id_list_for_batch),
#         property_id_list_for_batch,
#         flush=True,
#     )

#     print("Batch Concepts", len(batch[0]), flush=True)
#     print(batch[0], flush=True)
#     print("Batch Properties", len(batch[1]), flush=True)
#     print(batch[1], flush=True)

#     total_batch_loss = 0.0

#     for i in range(len(concept_id_list_for_batch)):

#         concept_id = concept_id_list_for_batch[i]

#         print(flush=True)
#         print(f"Processing concept ID : {i}, {concept_id}", flush=True)

#         # Extracting the property of the concept at the whole dataset level.
#         property_id_list_for_concept = torch.tensor(
#             dataset.con_pro_dict[concept_id.item()], device=device
#         )

#         # Extracting the negative property by excluding the properties that the concept may have at the  whole dataset level
#         negative_property_id_for_concept = torch.tensor(
#             [
#                 x
#                 for x in property_id_list_for_batch
#                 if x not in property_id_list_for_concept
#             ],
#             device=device,
#         )

#         print(f"property_id_list_for_batch : {property_id_list_for_batch}")
#         print(f"property_id_list_for_concept : {property_id_list_for_concept}")
#         print(f"negative_property_id_for_concept : {negative_property_id_for_concept}")

#         positive_property_for_concept_mask = torch.tensor(
#             [
#                 [1] if x in negative_property_id_for_concept else [0]
#                 for x in property_id_list_for_batch
#             ],
#             device=device,
#         )

#         positive_property_for_concept_mask[i] = 0

#         print(
#             f"i, positive_property_for_concept_mask : {i} , {positive_property_for_concept_mask}"
#         )

#         neg_property_embedding = torch.mul(
#             property_embedding, positive_property_for_concept_mask
#         )

#         concept_embed = concept_embedding[i].unsqueeze(0)
#         positive_property_embed = property_embedding[i].unsqueeze(0)

#         print(f"concept_embed.shape : {concept_embed.shape}")
#         print(f"positive_property_embed.shape : {positive_property_embed.shape}")
#         print(f"neg_property_embedding.shape : {neg_property_embedding.shape}")
#         print(f"neg_property_embedding")
#         print(neg_property_embedding)

#         batch_for_concept = torch.cat(
#             (concept_embed, positive_property_embed, neg_property_embedding), dim=0,
#         ).to(device)

#         label = torch.arange(0, batch_for_concept.size(0), dtype=int, device=device)
#         label[1] = 0

#         print("concept_embed", concept_embed.size(), flush=True)
#         print(
#             "positive_property_embed", positive_property_embed.size(), flush=True,
#         )
#         print("label", label.size(), label, flush=True)
#         print("batch_for_concept", batch_for_concept.size(), flush=True)

#         loss_for_concept = loss_fn(batch_for_concept, label)

#         print(f"Loss for concept : {loss_for_concept}", flush=True)
#         print(flush=True)

#         total_batch_loss += loss_for_concept

#     print(f"Total Loss for batch : {total_batch_loss}", flush=True)

#     return total_batch_loss


def calculate_cross_entropy_loss(
    dataset, batch, concept_embedding, property_embedding, loss_fn, device
):

    # self.concept2idx, self.idx2concept = self.create_concept_idx_dicts()
    # self.property2idx, self.idx2property = self.create_property_idx_dicts()

    # print ("con_pro_dict :", dataset.con_pro_dict, "\n")

    # print ("\t  num_neg_concept :", num_neg_concept, flush=True)

    batch_logits, batch_labels = [], []

    concept_id_list_for_batch = torch.tensor(
        [dataset.concept2idx[concept] for concept in batch[0]], device=device
    )
    property_id_list_for_batch = torch.tensor(
        [dataset.property2idx[prop] for prop in batch[1]], device=device
    )

    # print ("concept_id_list_for_batch :", concept_id_list_for_batch)
    # print ("property_id_list_for_batch :", property_id_list_for_batch)

    # neg_concept_list, neg_property_list = [], []

    logits_pos_concepts = (
        (concept_embedding * property_embedding)
        .sum(-1)
        .reshape(concept_embedding.shape[0], 1)
    )
    labels_pos_concepts = torch.ones_like(
        logits_pos_concepts, dtype=torch.float32, device=device
    )

    batch_logits.append(logits_pos_concepts.flatten())
    batch_labels.append(labels_pos_concepts.flatten())

    # print ("\nlogits_pos_concepts :", logits_pos_concepts)
    # print ("labels :", labels)

    loss_pos_concept = loss_fn(logits_pos_concepts, labels_pos_concepts)
    # print ("Loss positive concepts :", loss_pos_concept)

    loss_neg_concept = 0.0

    for i in range(len(concept_id_list_for_batch)):

        concept_id = concept_id_list_for_batch[i]

        # Extracting the property of the concept at the whole dataset level.
        property_id_list_for_concept = torch.tensor(
            dataset.con_pro_dict[concept_id.item()], device=device
        )

        # Extracting the negative property by excluding the properties that the concept may have at the  whole dataset level
        negative_property_id_for_concept = torch.tensor(
            [
                x
                for x in property_id_list_for_batch
                if x not in property_id_list_for_concept
            ],
            device=device,
        )

        positive_property_for_concept_mask = torch.tensor(
            [
                [1] if x in negative_property_id_for_concept else [0]
                for x in property_id_list_for_batch
            ],
            device=device,
        )

        neg_property_embedding = torch.mul(
            property_embedding, positive_property_for_concept_mask
        )

        concept_i_repeated = (
            concept_embedding[i].unsqueeze(0).repeat(concept_embedding.shape[0], 1)
        )

        logits_neg_concepts = (
            (concept_i_repeated * neg_property_embedding)
            .sum(-1)
            .reshape(concept_i_repeated.shape[0], 1)
        )

        labels_neg_concepts = torch.zeros_like(
            logits_neg_concepts, dtype=torch.float32, device=device
        )

        batch_logits.append(logits_neg_concepts.flatten())
        batch_labels.append(labels_neg_concepts.flatten())

        loss_neg_concept += loss_fn(logits_neg_concepts, labels_neg_concepts)

    batch_logits = torch.vstack(batch_logits).reshape(-1, 1)
    batch_labels = torch.vstack(batch_labels).reshape(-1, 1)

    return loss_pos_concept + loss_neg_concept, batch_logits, batch_labels


def load_pretrained_model(config):

    model = create_model(config.get("model_params"))
    pretrained_model_path = config["model_params"]["pretrained_model_path"]

    log.info(f"Loading the pretrained model loaded from : {pretrained_model_path}")

    model.load_state_dict(torch.load(pretrained_model_path))

    log.info(f"The pretrained model is loaded : {pretrained_model_path}")

    return model


def mcrae_dataset_and_dataloader(dataset_params, dataset_type, data_df=None):

    if dataset_type in ("train", "valid"):
        dataset = McRaeConceptPropertyDataset(
            dataset_params=dataset_params, dataset_type=dataset_type, data_df=data_df
        )
        data_sampler = RandomSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_params["loader_params"]["batch_size"],
            sampler=data_sampler,
            num_workers=dataset_params["loader_params"]["num_workers"],
            pin_memory=dataset_params["loader_params"]["pin_memory"],
        )

    elif dataset_type in ("test",):

        if data_df is not None:
            dataset = McRaeConceptPropertyDataset(
                dataset_params=dataset_params,
                dataset_type=dataset_type,
                data_df=data_df,
            )
        else:
            dataset = McRaeConceptPropertyDataset(
                dataset_params=dataset_params, dataset_type=dataset_type, data_df=None,
            )

        data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_params["loader_params"]["batch_size"],
            sampler=data_sampler,
            shuffle=False,
            num_workers=dataset_params["loader_params"]["num_workers"],
            pin_memory=dataset_params["loader_params"]["pin_memory"],
        )

    return dataset, dataloader


def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return (total_params, trainable_params)


def plot_loss_curve(train_loss, val_loss, export_path=None, plot_name="loss_curve.png"):

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    epochs = range(1, len(train_loss) + 1)

    print(len(train_loss))
    print(len(val_loss))

    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(range(0, len(train_loss) + 1, 1))
    plt.legend(loc="best")
    plt.grid(True)

    if export_path:
        plot_name = os.path.join(export_path, plot_name)

    plt.savefig(plot_name)

    plt.show()
    plt.close()
