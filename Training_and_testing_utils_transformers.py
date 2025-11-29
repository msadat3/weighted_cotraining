import copy
import os.path as p
from statistics import mean

import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from Data_utils import *
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, TensorDataset

from Models import *


# copied from: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def reduce_loss(self, loss):
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum() if self.reduction == "sum" else loss
        )

    def forward(self, inputs, targets, is_target_onehot=False):
        assert 0 <= self.smoothing < 1

        if is_target_onehot == False:
            targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


def get_initial_weights(
    location_1,
    location_2,
    model_type,
    X_train_1,
    att_mask_train_1,
    y_train_1,
    X_train_2,
    att_mask_train_2,
    y_train_2,
    X_valid,
    att_mask_valid,
    y_valid,
    X_autoLabeled,
    att_mask_autoLabeled,
    y_autoLabeled,
    device_1,
    device_2,
    batch_size,
    accumulation_steps,
    num_epochs,
    num_classes,
    report_every,
    epoch_patience,
    load=False,
    dropout=0,
    lr=2e-5,
):
    """
    Compute initial weights for automatically labeled samples by training two models on human annotated data.
    """

    train_probabilities_all_epochs_1 = [[] for _ in range(len(X_autoLabeled))]
    train_probabilities_all_epochs_2 = [[] for _ in range(len(X_autoLabeled))]

    x_tr_1 = torch.tensor(X_train_1, dtype=torch.long)
    att_mask_tr_1 = torch.tensor(att_mask_train_1, dtype=torch.long)
    y_tr_1 = torch.tensor(y_train_1, dtype=torch.long)

    x_tr_2 = torch.tensor(X_train_2, dtype=torch.long)
    att_mask_tr_2 = torch.tensor(att_mask_train_2, dtype=torch.long)
    y_tr_2 = torch.tensor(y_train_2, dtype=torch.long)

    x_val = torch.tensor(X_valid, dtype=torch.long)
    att_mask_val = torch.tensor(att_mask_valid, dtype=torch.long)
    y_val = torch.tensor(y_valid, dtype=torch.long)

    x_auto = torch.tensor(X_autoLabeled, dtype=torch.long)
    att_mask_auto = torch.tensor(att_mask_autoLabeled, dtype=torch.long)
    y_auto = torch.tensor(y_autoLabeled, dtype=torch.long)

    train = TensorDataset(x_tr_1, y_tr_1, att_mask_tr_1, x_tr_2, y_tr_2, att_mask_tr_2)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)

    val = TensorDataset(x_val, y_val, att_mask_val)
    validationLoader = DataLoader(val, batch_size=batch_size)

    autoDataset = TensorDataset(x_auto, y_auto, att_mask_auto)  # renamed
    autoLoader = DataLoader(autoDataset, batch_size=batch_size)

    best_checkpoint_epoch = 0

    # model loading
    if model_type == "Sci_BERT":
        model_1 = Sci_BERT(num_classes)
        model_2 = Sci_BERT(num_classes)
    elif model_type == "RoBERTa":
        model_1 = RoBERTa(num_classes)
        model_2 = RoBERTa(num_classes)

    model_1.to(torch.device(device_1))
    model_2.to(torch.device(device_2))

    if load:
        model_1.load_state_dict(torch.load(location_1 + "/model.pt"))
        model_2.load_state_dict(torch.load(location_2 + "/model.pt"))

    optimizer_1 = optim.Adam(model_1.parameters(), lr=lr)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss(reduction="none")

    prev_best_score = -1
    notImprovingWeights_epoch = 0

    for epoch in range(num_epochs):
        if notImprovingWeights_epoch == epoch_patience:
            print("Performance not improving for 5 consecutive epochs.")
            break

        model_1.train()
        model_2.train()

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        i = 0

        for data_1, target_1, att_1, data_2, target_2, att_2 in trainloader:
            data_1 = data_1.to(device_1)
            target_1 = target_1.to(device_1)
            att_1 = att_1.to(device_1)

            data_2 = data_2.to(device_2)
            target_2 = target_2.to(device_2)
            att_2 = att_2.to(device_2)

            out_1 = model_1(data_1, att_1)
            out_2 = model_2(data_2, att_2)

            loss_1 = criterion(out_1, target_1).mean() / accumulation_steps
            loss_2 = criterion(out_2, target_2).mean() / accumulation_steps

            loss_1.backward()
            loss_2.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), 1)
                optimizer_1.step()
                optimizer_2.step()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
            i += 1

        # evaluation on auto-labeled
        model_1.eval()
        model_2.eval()

        curr_batch = 0
        with torch.no_grad():
            for auto_data, auto_target, auto_att in autoLoader:

                auto_data_1 = auto_data.to(device_1)
                auto_target_1 = auto_target.to(device_1)
                auto_att_1 = auto_att.to(device_1)

                auto_data_2 = auto_data.to(device_2)
                auto_target_2 = auto_target.to(device_2)
                auto_att_2 = auto_att.to(device_2)

                out_1 = model_1(auto_data_1, auto_att_1)
                prob_1 = torch.softmax(out_1, dim=1).cpu().numpy().tolist()

                out_2 = model_2(auto_data_2, auto_att_2)
                prob_2 = torch.softmax(out_2, dim=1).cpu().numpy().tolist()

                golds = auto_target_1.cpu().numpy().tolist()

                for idx in range(len(golds)):
                    train_probabilities_all_epochs_1[
                        idx + curr_batch * batch_size
                    ].append(prob_1[idx][golds[idx]])
                    train_probabilities_all_epochs_2[
                        idx + curr_batch * batch_size
                    ].append(prob_2[idx][golds[idx]])

                curr_batch += 1

        # validation
        val_out = []
        val_out_1 = []
        val_out_2 = []
        for val_data, val_target, val_att in validationLoader:
            val_data_1 = val_data.to(device_1)
            val_att_1 = val_att.to(device_1)
            out1 = model_1(val_data_1, val_att_1)
            prob1 = torch.softmax(out1, dim=1)

            val_data_2 = val_data.to(device_2)
            val_att_2 = val_att.to(device_2)
            out2 = model_2(val_data_2, val_att_2)
            prob2 = torch.softmax(out2, dim=1)

            val_out_1 += torch.argmax(prob1, dim=1).cpu().numpy().tolist()
            val_out_2 += torch.argmax(prob2, dim=1).cpu().numpy().tolist()

            summed = prob1.cpu() + prob2.cpu()
            val_out += torch.argmax(summed, dim=1).numpy().tolist()

        current_score_1 = f1_score(y_valid.tolist(), val_out_1, average="macro")
        current_score_2 = f1_score(y_valid.tolist(), val_out_2, average="macro")
        current_score = f1_score(y_valid.tolist(), val_out, average="macro")

        if current_score > prev_best_score:
            best_checkpoint_epoch = epoch
            prev_best_score = current_score
            notImprovingWeights_epoch = 0

            torch.save(model_1.state_dict(), location_1 + "/model.pt")
            torch.save(model_2.state_dict(), location_2 + "/model.pt")
        else:
            notImprovingWeights_epoch += 1

    # compute weights
    auto_weights_1 = []
    auto_weights_2 = []

    best_epoch_probs_1 = []
    best_epoch_probs_2 = []

    for i in range(len(train_probabilities_all_epochs_1)):
        best_epoch_probs_1.append(
            train_probabilities_all_epochs_1[i][best_checkpoint_epoch]
        )
        best_epoch_probs_2.append(
            train_probabilities_all_epochs_2[i][best_checkpoint_epoch]
        )

        max_epoch = max(best_checkpoint_epoch, 1)

        c1 = mean(train_probabilities_all_epochs_1[i][: max_epoch + 1])
        c2 = mean(train_probabilities_all_epochs_2[i][: max_epoch + 1])

        v1 = np.std(train_probabilities_all_epochs_1[i][: max_epoch + 1])
        v2 = np.std(train_probabilities_all_epochs_2[i][: max_epoch + 1])

        auto_weights_1.append(c2 - v2)
        auto_weights_2.append(c1 + v1)

    auto_weights_norm_1 = max_min_normalize_all_values(auto_weights_1)
    auto_weights_norm_2 = max_min_normalize_all_values(auto_weights_2)

    return (
        auto_weights_1,
        auto_weights_2,
        auto_weights_norm_1,
        auto_weights_norm_2,
        train_probabilities_all_epochs_1,
        train_probabilities_all_epochs_2,
        best_epoch_probs_1,
        best_epoch_probs_2,
    )


def weighted_co_training(
    location_1,
    location_2,
    model_type,
    X_train,
    att_mask_train,
    y_train,
    initial_train_weights_1,
    initial_train_weights_2,
    initial_probabilities_1,
    initial_probabilities_2,
    X_valid,
    att_mask_valid,
    y_valid,
    device_1,
    device_2,
    batch_size,
    accumulation_steps,
    num_epochs,
    num_classes,
    report_every,
    epoch_patience,
    lr=2e-5,
):

    train_weights_1 = np.array(initial_train_weights_1)
    train_weights_2 = np.array(initial_train_weights_2)

    train_weights_1_raw = copy.deepcopy(initial_train_weights_1)
    train_weights_2_raw = copy.deepcopy(initial_train_weights_2)

    train_probabilities_all_epochs_1 = [
        [initial_probabilities_1[i]] for i in range(len(initial_probabilities_1))
    ]
    train_probabilities_all_epochs_2 = [
        [initial_probabilities_2[i]] for i in range(len(initial_probabilities_2))
    ]

    best_epoch_probabilities_1 = [0] * len(train_probabilities_all_epochs_1)
    best_epoch_probabilities_2 = [0] * len(train_probabilities_all_epochs_1)

    best_train_weights_1 = copy.deepcopy(initial_train_weights_1)
    best_train_weights_2 = copy.deepcopy(initial_train_weights_2)

    best_train_weights_1_raw = copy.deepcopy(initial_train_weights_1)
    best_train_weights_2_raw = copy.deepcopy(initial_train_weights_2)

    x_tr = torch.tensor(X_train, dtype=torch.long)
    att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    id_tr = [i for i in range(len(x_tr))]  # can also be used as index
    id_tr = torch.tensor(id_tr, dtype=torch.long)

    x_val = torch.tensor(X_valid, dtype=torch.long)
    att_mask_val = torch.tensor(att_mask_valid, dtype=torch.long)
    y_val = torch.tensor(y_valid, dtype=torch.long)

    train = TensorDataset(id_tr, x_tr, y_tr, att_mask_tr)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = TensorDataset(x_val, y_val, att_mask_val)
    validationLoader = DataLoader(val, batch_size=batch_size)

    best_checkpoint_epoch = 0

    if model_type == "Sci_BERT":
        model_1 = Sci_BERT(num_classes)
        model_2 = Sci_BERT(num_classes)
    elif model_type == "RoBERTa":
        model_1 = RoBERTa(num_classes)
        model_2 = RoBERTa(num_classes)

    model_1.to(torch.device(device_1))
    model_2.to(torch.device(device_2))

    optimizer_1 = optim.Adam(model_1.parameters(), lr=lr)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=lr)
    criterion = SmoothCrossEntropyLoss(reduction="none")

    prev_best_score = -1
    notImprovingWeights_epoch = 0

    for epoch in range(0, num_epochs):

        train_weights_new_1 = copy.deepcopy(train_weights_1)
        train_weights_new_2 = copy.deepcopy(train_weights_2)

        if notImprovingWeights_epoch == epoch_patience:
            print(f"Performance not improving for {epoch_patience} consecutive epochs.")
            break

        model_1.train()
        model_2.train()

        i = 0
        step_count = 0
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        for tr_ids, data, target, att in trainloader:
            data_1 = data.to(torch.device(device_1))
            target_1 = target.to(torch.device(device_1))
            att_1 = att.to(torch.device(device_1))
            weights_1 = torch.tensor(
                train_weights_1[tr_ids.tolist()], dtype=torch.double
            ).to(torch.device(device_1))

            data_2 = data.to(torch.device(device_2))
            target_2 = target.to(torch.device(device_2))
            att_2 = att.to(torch.device(device_2))
            weights_2 = torch.tensor(
                train_weights_2[tr_ids.tolist()], dtype=torch.double
            ).to(torch.device(device_2))

            output_1 = model_1(data_1, att_1)
            output_2 = model_2(data_2, att_2)

            loss_1 = criterion(output_1, target_1)
            loss_2 = criterion(output_2, target_2)

            loss_1 = loss_1 * weights_1
            loss_2 = loss_2 * weights_2

            loss_1 = torch.mean(loss_1)
            loss_2 = torch.mean(loss_2)

            loss_1 = loss_1 / accumulation_steps
            loss_2 = loss_2 / accumulation_steps
            loss_1.backward()
            loss_2.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), 1)
                optimizer_1.step()
                optimizer_1.zero_grad()

                optimizer_2.step()
                optimizer_2.zero_grad()

                step_count += 1
            i += 1

            with torch.no_grad():
                probabilities_1 = torch.softmax(output_1, dim=1)
                probabilities_2 = torch.softmax(output_2, dim=1)
                probabilities_1 = probabilities_1.cpu().detach().numpy().tolist()
                probabilities_2 = probabilities_2.cpu().detach().numpy().tolist()
                golds = target_1.cpu().detach().numpy().tolist()
                tr_ids = tr_ids.tolist()

                for idx, tr_id in enumerate(tr_ids):
                    train_probabilities_all_epochs_1[tr_id].append(
                        probabilities_1[idx][golds[idx]]
                    )
                    train_probabilities_all_epochs_2[tr_id].append(
                        probabilities_2[idx][golds[idx]]
                    )

                    c1 = mean(train_probabilities_all_epochs_1[tr_id])
                    c2 = mean(train_probabilities_all_epochs_2[tr_id])

                    v1 = np.std(train_probabilities_all_epochs_1[tr_id])
                    v2 = np.std(train_probabilities_all_epochs_2[tr_id])

                    train_weights_new_1[tr_id] = c2 - v2
                    train_weights_new_2[tr_id] = c1 + v1

        train_weights_new_normalized_1 = max_min_normalize_all_values(
            copy.deepcopy(train_weights_new_1)
        )
        train_weights_new_normalized_2 = max_min_normalize_all_values(
            copy.deepcopy(train_weights_new_2)
        )

        model_1.eval()
        model_2.eval()

        n = 0
        print("============= Epoch " + str(epoch) + " =============")
        with torch.no_grad():
            val_out_1 = []
            val_out_2 = []
            val_ensembled_out = []
            for val_data, val_target, att in validationLoader:
                val_data_1 = val_data.to(torch.device(device_1))
                val_target_1 = val_target.to(torch.device(device_1))
                att_1 = att.to(torch.device(device_1))

                val_data_2 = val_data.to(torch.device(device_2))
                val_target_2 = val_target.to(torch.device(device_2))
                att_2 = att.to(torch.device(device_2))

                out_1 = model_1(val_data_1, att_1)
                val_probabilities_1 = torch.softmax(out_1, dim=1)
                out_1 = torch.argmax(out_1, dim=1)
                out_1 = out_1.cpu().detach().numpy()
                val_out_1 += out_1.tolist()

                out_2 = model_2(val_data_2, att_2)
                val_probabilities_2 = torch.softmax(out_2, dim=1)
                out_2 = torch.argmax(out_2, dim=1)
                out_2 = out_2.cpu().detach().numpy()
                val_out_2 += out_2.tolist()

                val_probabilities_summed = (
                    val_probabilities_1.cpu() + val_probabilities_2.cpu()
                )

                out_ensembled = torch.argmax(val_probabilities_summed, dim=1)

                val_ensembled_out += out_ensembled.tolist()

                n += len(val_target_1)

            current_score = f1_score(
                y_valid.tolist(), val_ensembled_out, average="macro"
            )

            current_score_1 = f1_score(y_valid.tolist(), val_out_1, average="macro")
            current_score_2 = f1_score(y_valid.tolist(), val_out_2, average="macro")

            if current_score > prev_best_score:
                best_checkpoint_epoch = epoch
                print(
                    "Validation f1 score improved from",
                    prev_best_score,
                    "to",
                    current_score,
                    "saving model...",
                )
                prev_best_score = current_score
                last_checkpoint_info = {"epoch": epoch, "score": prev_best_score}
                save_data(last_checkpoint_info, location_1 + "/checkpoint.pkl")
                torch.save(model_1.state_dict(), location_1 + "/model.pt")
                torch.save(model_2.state_dict(), location_2 + "/model.pt")

                print("current_score_1", current_score_1)
                print("current_score_2", current_score_2)
                with open(location_1 + "/checkpoint.txt", "w") as file:
                    file.write("epoch:" + str(epoch))
                    file.write("current_score_1:" + str(current_score_1))

                with open(location_2 + "/checkpoint.txt", "w") as file:
                    file.write("epoch:" + str(epoch))
                    file.write("current_score_2:" + str(current_score_2))

                for b_idx in range(len(best_epoch_probabilities_1)):
                    best_epoch_probabilities_1[b_idx] = (
                        train_probabilities_all_epochs_1[b_idx][-1]
                    )
                    best_epoch_probabilities_2[b_idx] = (
                        train_probabilities_all_epochs_2[b_idx][-1]
                    )

                best_train_weights_1 = copy.deepcopy(train_weights_1)
                best_train_weights_2 = copy.deepcopy(train_weights_2)

                best_train_weights_1_raw = copy.deepcopy(train_weights_1_raw)
                best_train_weights_2_raw = copy.deepcopy(train_weights_2_raw)

                notImprovingWeights_epoch = 0
            else:
                print("Validation f1 score did not improve from", prev_best_score)
                notImprovingWeights_epoch += 1

            train_weights_1 = np.array(copy.deepcopy(train_weights_new_normalized_1))
            train_weights_2 = np.array(copy.deepcopy(train_weights_new_normalized_2))

            train_weights_1_raw = copy.deepcopy(train_weights_new_1)
            train_weights_2_raw = copy.deepcopy(train_weights_new_2)

    return (
        best_train_weights_1,
        best_train_weights_2,
        best_train_weights_1_raw,
        best_train_weights_2_raw,
        train_probabilities_all_epochs_1,
        train_probabilities_all_epochs_2,
        best_epoch_probabilities_1,
        best_epoch_probabilities_2,
    )


def finetune_on_human_labeled(
    location_1,
    location_2,
    load_location_1,
    load_location_2,
    model_type,
    X_train_1,
    att_mask_train_1,
    y_train_1,
    X_train_2,
    att_mask_train_2,
    y_train_2,
    X_valid,
    att_mask_valid,
    y_valid,
    device_1,
    device_2,
    batch_size,
    accumulation_steps,
    num_epochs,
    num_classes,
    epoch_patience,
    load=False,
    dropout=0,
    lr=2e-5,
):

    x_tr_1 = torch.tensor(X_train_1, dtype=torch.long)
    att_mask_tr_1 = torch.tensor(att_mask_train_1, dtype=torch.long)
    y_tr_1 = torch.tensor(y_train_1, dtype=torch.long)

    x_tr_2 = torch.tensor(X_train_2, dtype=torch.long)
    att_mask_tr_2 = torch.tensor(att_mask_train_2, dtype=torch.long)
    y_tr_2 = torch.tensor(y_train_2, dtype=torch.long)

    x_val = torch.tensor(X_valid, dtype=torch.long)
    att_mask_val = torch.tensor(att_mask_valid, dtype=torch.long)
    y_val = torch.tensor(y_valid, dtype=torch.long)

    train = TensorDataset(x_tr_1, y_tr_1, att_mask_tr_1, x_tr_2, y_tr_2, att_mask_tr_2)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)

    val = TensorDataset(x_val, y_val, att_mask_val)
    validationLoader = DataLoader(val, batch_size=batch_size)

    best_checkpoint_epoch = 0

    if model_type == "Sci_BERT":
        model_1 = Sci_BERT(num_classes)
        model_2 = Sci_BERT(num_classes)
    elif model_type == "RoBERTa":
        model_1 = RoBERTa(num_classes)
        model_2 = RoBERTa(num_classes)

    model_1.to(torch.device(device_1))
    model_2.to(torch.device(device_2))

    if load == True:
        model_1.load_state_dict(torch.load(load_location_1 + "/model.pt"))
        model_2.load_state_dict(torch.load(load_location_2 + "/model.pt"))

    optimizer_1 = optim.Adam(model_1.parameters(), lr=lr)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss(reduction="none")

    prev_best_score = -1
    notImprovingWeights_epoch = 0

    for epoch in range(0, num_epochs):
        if notImprovingWeights_epoch == epoch_patience:
            print("Performance not improving for 5 consecutive epochs.")
            break
        model_1.train()
        model_2.train()

        i = 0
        step_count = 0
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        for data_1, target_1, att_1, data_2, target_2, att_2 in trainloader:
            data_1 = data_1.to(torch.device(device_1))
            target_1 = target_1.to(torch.device(device_1))
            att_1 = att_1.to(torch.device(device_1))

            data_2 = data_2.to(torch.device(device_2))
            target_2 = target_2.to(torch.device(device_2))
            att_2 = att_2.to(torch.device(device_2))

            output_1 = model_1(data_1, att_1)
            output_2 = model_2(data_2, att_2)

            loss_1 = criterion(output_1, target_1)
            loss_2 = criterion(output_2, target_2)

            loss_1 = torch.mean(loss_1)
            loss_2 = torch.mean(loss_2)

            loss_1 = loss_1 / accumulation_steps
            loss_2 = loss_2 / accumulation_steps
            loss_1.backward()
            loss_2.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), 1)
                optimizer_1.step()
                optimizer_1.zero_grad()

                optimizer_2.step()
                optimizer_2.zero_grad()

                step_count += 1
            i += 1

        model_1.eval()
        model_2.eval()

        n = 0
        print("=============Epoch " + str(epoch) + " =============")
        with torch.no_grad():

            val_out_1 = []
            val_out_2 = []
            val_out = []
            for val_data, val_target, val_att in validationLoader:
                val_data_1 = val_data.to(torch.device(device_1))
                val_target_1 = val_target.to(torch.device(device_1))
                val_att_1 = val_att.to(torch.device(device_1))
                out_1 = model_1(val_data_1, val_att_1)
                val_probabilities_1 = torch.softmax(out_1, dim=1)

                val_data_2 = val_data.to(torch.device(device_2))
                val_target_2 = val_target.to(torch.device(device_2))
                val_att_2 = val_att.to(torch.device(device_2))
                out_2 = model_2(val_data_2, val_att_2)
                val_probabilities_2 = torch.softmax(out_2, dim=1)

                out_1 = torch.argmax(val_probabilities_1, dim=1)
                out_2 = torch.argmax(val_probabilities_2, dim=1)

                val_probabilities_summed = (
                    val_probabilities_1.cpu() + val_probabilities_2.cpu()
                )

                out_ensembled = torch.argmax(val_probabilities_summed, dim=1)

                out_ensembled = out_ensembled.cpu().detach().numpy()

                val_out += out_ensembled.tolist()
                val_out_1 += out_1.cpu().detach().numpy().tolist()
                val_out_2 += out_2.cpu().detach().numpy().tolist()
                # print('val out shape', val_out.shape)
                n += len(val_target)

            current_score_1 = f1_score(y_valid.tolist(), val_out_1, average="macro")
            current_score_2 = f1_score(y_valid.tolist(), val_out_2, average="macro")

            current_score = f1_score(y_valid.tolist(), val_out, average="macro")
            # print(scores)
            # print(best_thresholds)
            if current_score > prev_best_score:

                print(
                    "Validation f1 score improved from",
                    prev_best_score,
                    "to",
                    current_score,
                    "saving model...",
                )
                print("current_score_1", current_score_1)
                print("current_score_2", current_score_2)
                prev_best_score = current_score

                with open(location_1 + "/checkpoint.txt", "w") as file:
                    file.write("epoch:" + str(epoch))
                    file.write("current_score_1:" + str(current_score_1))

                with open(location_2 + "/checkpoint.txt", "w") as file:
                    file.write("epoch:" + str(epoch))
                    file.write("current_score_2:" + str(current_score_2))

                torch.save(model_1.state_dict(), location_1 + "/model.pt")
                torch.save(model_2.state_dict(), location_2 + "/model.pt")
                notImprovingWeights_epoch = 0
            else:
                print("Validation f1 score did not improve from", prev_best_score)
                print("current_score", current_score)
                print("current_score_1", current_score_1)
                print("current_score_2", current_score_2)
                notImprovingWeights_epoch += 1


def test_model_ensemble(
    location_1,
    location_2,
    model_type,
    x,
    att_x,
    y,
    device,
    batch_size,
    num_classes,
    print_res,
    save_loc,
):
    if model_type == "Sci_BERT":
        test_model_1 = Sci_BERT(num_classes)
        test_model_2 = Sci_BERT(num_classes)
    elif model_type == "RoBERTa":
        test_model_1 = RoBERTa(num_classes)
        test_model_2 = RoBERTa(num_classes)

    test_model_1.to(torch.device(device))
    test_model_2.to(torch.device(device))

    test_model_1.load_state_dict(
        torch.load(location_1 + "/model.pt", map_location="cuda:0")
    )
    test_model_2.load_state_dict(
        torch.load(location_2 + "/model.pt", map_location="cuda:0")
    )

    test_model_1.eval()
    test_model_2.eval()

    x_te = torch.tensor(x, dtype=torch.long)
    att_mask_te = torch.tensor(att_x, dtype=torch.long)
    y_te = torch.tensor(y, dtype=torch.long)
    te = TensorDataset(x_te, y_te, att_mask_te)
    testLoader = DataLoader(te, batch_size=batch_size)

    with torch.no_grad():
        test_out = []
        for test_data, test_target, att in testLoader:
            test_data = test_data.to(torch.device(device))
            test_target = test_target.to(torch.device(device))
            att = att.to(torch.device(device))

            out_1 = test_model_1(test_data, att)
            out_2 = test_model_2(test_data, att)

            probabilities_1 = torch.softmax(out_1, dim=1)
            probabilities_2 = torch.softmax(out_2, dim=1)

            probabilities_summed = probabilities_1 + probabilities_2

            out = torch.argmax(probabilities_summed, dim=1)

            out = out.cpu().detach().numpy()
            test_out += out.tolist()

        current_score = f1_score(y.tolist(), test_out, average="macro")
        Accuracy = accuracy_score(y.tolist(), test_out)

        print("Macro f1", current_score)
        print("Accuracy", Accuracy)

        performance = classification_report(
            y.tolist(),
            test_out,
            target_names=["contrasting", "reasoning", "entailing", "neutral"],
            digits=4,
            output_dict=True,
        )

        performance_df = pandas.DataFrame(performance).transpose()

        performance_df.to_csv(save_loc)

        return current_score


def max_min_normalize(val, max_val, min_val):
    return (val - min_val) / (max_val - min_val)


def max_min_normalize_all_values(items):

    max_item = max(items)
    min_item = min(items)

    items_normalized = [max_min_normalize(item, max_item, min_item) for item in items]

    return items_normalized
