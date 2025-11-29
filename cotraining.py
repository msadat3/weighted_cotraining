import argparse
import os
import os.path as p
import random

import numpy as np
import pandas
import torch
from transformers import *

from Data_utils import *
from Training_and_testing_utils_transformers import *

label_to_idx_dict = {
    "contrasting": 0,
    "reasoning": 1,
    "entailment": 2,
    "neutral": 3,
}

idx_to_label_dict = {v: k for k, v in label_to_idx_dict.items()}


def get_numeric_label(label):
    if label in label_to_idx_dict:
        return label_to_idx_dict[label]
    else:
        print(f"Unknown label: {label}")
        return -1


def get_textual_label(label):
    if label in idx_to_label_dict.keys():
        return idx_to_label_dict[label]
    else:
        return -1


def parse_args():
    parser = argparse.ArgumentParser(description="Co-training for NLI")

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)",
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="RoBERTa",
        choices=["RoBERTa", "Sci_BERT"],
        help="Model type to use (default: RoBERTa)",
    )

    # Paths
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base location for the experiment. Should contain train, test, dev files and iteration_0 subfolder",
    )

    parser.add_argument(
        "--dataset", type=str, default="SciNLI", help="Dataset name (default: SciNLI)"
    )

    # Training parameters

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Patience for early stopping (default: 2)",
    )

    parser.add_argument(
        "--epoch_patience", type=int, default=2, help="Patience for epochs (default: 2)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )

    parser.add_argument(
        "--report_every",
        type=int,
        default=10,
        help="Report training progress every N steps (default: 10)",
    )

    # GPU configuration
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1",
        help='CUDA visible devices (default: "0,1")',
    )

    parser.add_argument(
        "--device_1",
        type=str,
        default="cuda:0",
        help="First device for training (default: cuda:0)",
    )

    parser.add_argument(
        "--device_2",
        type=str,
        default="cuda:1",
        help="Second device for training (default: cuda:1)",
    )

    # Learning rates
    parser.add_argument(
        "--lr_initial",
        type=float,
        default=2e-5,
        help="Learning rate for initial training (default: 2e-5)",
    )

    parser.add_argument(
        "--lr_finetune",
        type=float,
        default=2e-6,
        help="Learning rate for finetuning (default: 2e-6)",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Calculate accumulation steps (to maintain effective batch size of 64)
    accumulation_steps = 64 // args.batch_size

    # Set number of classes
    num_classes = 4

    print(f"Starting experiment with the following configuration:")
    print(f"  Seed: {args.seed}")
    print(f"  Model: {args.model_type}")
    print(f"  Base path: {args.base}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Devices: {args.device_1}, {args.device_2}")
    print("=" * 66)

    # Load human-labeled datasets
    train_1 = pandas.read_csv(p.join(args.base, "train_1.tsv"), sep="\t")
    train_2 = pandas.read_csv(p.join(args.base, "train_2.tsv"), sep="\t")
    test_set = pandas.read_csv(p.join(args.base, "test.tsv"), sep="\t")
    valid_set = pandas.read_csv(p.join(args.base, "dev.tsv"), sep="\t")

    # Convert labels to numeric
    for df in [train_1, train_2, test_set, valid_set]:
        df["label"] = df["label"].apply(get_numeric_label)

    # Keep only necessary columns
    train_1 = train_1[["sentence1", "sentence2", "label"]]
    train_2 = train_2[["sentence1", "sentence2", "label"]]
    test_set = test_set[["sentence1", "sentence2", "label"]]
    valid_set = valid_set[["sentence1", "sentence2", "label"]]

    # Load tokenizer
    if args.model_type == "Sci_BERT":
        tokenizer = BertTokenizer.from_pretrained(
            "allenai/scibert_scivocab_cased", do_lower_case=False
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", do_lower_case=False
        )

    # Prepare human-labeled data for transformers
    create_data_for_transformers(
        args.base, label_to_idx_dict, train_1, tokenizer, "train_1", args.model_type
    )
    create_data_for_transformers(
        args.base, label_to_idx_dict, train_2, tokenizer, "train_2", args.model_type
    )
    create_data_for_transformers(
        args.base, label_to_idx_dict, test_set, tokenizer, "test", args.model_type
    )
    create_data_for_transformers(
        args.base, label_to_idx_dict, valid_set, tokenizer, "valid", args.model_type
    )

    # Load prepared human-labeled data
    X_train_1 = np.asarray(load_data(p.join(args.base, "X_train_1.pkl")))
    att_mask_train_1 = load_data(p.join(args.base, "att_mask_train_1.pkl"))
    y_train_1 = np.asarray(load_data(p.join(args.base, "y_train_1.pkl")))

    X_train_2 = np.asarray(load_data(p.join(args.base, "X_train_2.pkl")))
    att_mask_train_2 = load_data(p.join(args.base, "att_mask_train_2.pkl"))
    y_train_2 = np.asarray(load_data(p.join(args.base, "y_train_2.pkl")))

    X_test = np.asarray(load_data(p.join(args.base, "X_test.pkl")))
    att_mask_test = load_data(p.join(args.base, "att_mask_test.pkl"))
    y_test = np.asarray(load_data(p.join(args.base, "y_test.pkl")))

    X_valid = np.asarray(load_data(p.join(args.base, "X_valid.pkl")))
    att_mask_valid = load_data(p.join(args.base, "att_mask_valid.pkl"))
    y_valid = np.asarray(load_data(p.join(args.base, "y_valid.pkl")))

    # ----------------------------
    # Load auto-labeled data
    # ----------------------------
    autoLabeledSet = pandas.read_csv(
        p.join(args.base, "Automatically_annotated.tsv"), sep="\t"
    )
    autoLabeledSet = autoLabeledSet[["sentence1", "sentence2", "label"]]
    autoLabeledSet["label"] = autoLabeledSet["label"].apply(get_numeric_label)

    create_data_for_transformers(
        args.base,
        label_to_idx_dict,
        autoLabeledSet,
        tokenizer,
        "autoLabeled",
        args.model_type,
    )

    X_autoLabeled = np.asarray(load_data(p.join(args.base, "X_autoLabeled.pkl")))
    att_mask_autoLabeled = load_data(p.join(args.base, "att_mask_autoLabeled.pkl"))
    y_autoLabeled = np.asarray(load_data(p.join(args.base, "y_autoLabeled.pkl")))

    print(
        "Data prepared and loaded"
    )

    # ----------------------------
    # Compute initial weights
    # ----------------------------
    (
        W_autoLabeled_raw_1,
        W_autoLabeled_raw_2,
        W_autoLabeled_1,
        W_autoLabeled_2,
        probabilities_all_epochs_1,
        probabilities_all_epochs_2,
        probabilities_best_epoch_1,
        probabilities_best_epoch_2,
    ) = get_initial_weights(
        location_1=args.base + "model_1/",
        location_2=args.base + "model_2/",
        model_type=args.model_type,
        X_train_1=X_train_1,
        att_mask_train_1=att_mask_train_1,
        y_train_1=y_train_1,
        X_train_2=X_train_2,
        att_mask_train_2=att_mask_train_2,
        y_train_2=y_train_2,
        X_valid=X_valid,
        att_mask_valid=att_mask_valid,
        y_valid=y_valid,
        X_autoLabeled=X_autoLabeled,
        att_mask_autoLabeled=att_mask_autoLabeled,
        y_autoLabeled=y_autoLabeled,
        device_1=args.device_1,
        device_2=args.device_2,
        batch_size=args.batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=args.num_epochs,
        num_classes=num_classes,
        report_every=args.report_every,
        epoch_patience=args.epoch_patience,
        load=False,
        dropout=0,
        lr=args.lr_initial,
    )

    # Save weights in autoLabeledSet
    autoLabeledSet["weight_1"] = W_autoLabeled_1
    autoLabeledSet["weight_2"] = W_autoLabeled_2
    autoLabeledSet["weight_raw_1"] = W_autoLabeled_raw_1
    autoLabeledSet["weight_raw_2"] = W_autoLabeled_raw_2
    autoLabeledSet["probabilities_all_epochs_1"] = probabilities_all_epochs_1
    autoLabeledSet["probabilities_all_epochs_2"] = probabilities_all_epochs_2
    autoLabeledSet["probabilities_best_epoch_1"] = probabilities_best_epoch_1
    autoLabeledSet["probabilities_best_epoch_2"] = probabilities_best_epoch_2

    # ----------------------------
    # Co-training
    # ----------------------------
    _ = weighted_co_training(
        model_1_location=args.base + "model_1/",
        model_2_location=args.base + "model_2/",
        model_type=args.model_type,
        X_pretrain=X_autoLabeled,
        att_mask_pretrain=att_mask_autoLabeled,
        y_pretrain=y_autoLabeled,
        W_pretrain_1=W_autoLabeled_1,
        W_pretrain_2=W_autoLabeled_2,
        initial_probabilities_1=probabilities_best_epoch_1,
        initial_probabilities_2=probabilities_best_epoch_2,
        X_valid=X_valid,
        att_mask_valid=att_mask_valid,
        y_valid=y_valid,
        device_1=args.device_1,
        device_2=args.device_2,
        batch_size=args.batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=args.num_epochs,
        num_classes=num_classes,
        report_every=args.report_every,
        epoch_patience=args.epoch_patience,
        lr=args.lr_initial,
    )

    # ----------------------------
    # Fine-tune on human-labeled data
    # ----------------------------
    finetune_on_human_labeled(
        location_1=args.base + "model_1_ft_hl/",
        location_2=args.base + "model_2_ft_hl/",
        load_location_1=args.base + "model_1/",
        load_location_2=args.base + "model_2/",
        model_type=args.model_type,
        X_train_1=X_train_1,
        att_mask_train_1=att_mask_train_1,
        y_train_1=y_train_1,
        X_train_2=X_train_2,
        att_mask_train_2=att_mask_train_2,
        y_train_2=y_train_2,
        X_valid=X_valid,
        att_mask_valid=att_mask_valid,
        y_valid=y_valid,
        device_1=args.device_1,
        device_2=args.device_2,
        batch_size=args.batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=args.num_epochs,
        num_classes=num_classes,
        epoch_patience=args.epoch_patience,
        load=True,
        dropout=0,
        lr=args.lr_finetune,
    )

    # ----------------------------
    # Evaluate models
    # ----------------------------
    test_score = test_model_ensemble(
        model_1_location=args.base + "model_1_ft_hl/",
        model_2_location=args.base + "model_2_ft_hl/",
        model_type=args.model_type,
        X_test=X_test,
        att_mask_test=att_mask_test,
        y_test=y_test,
        device=args.device_1,
        batch_size=args.batch_size,
        num_classes=num_classes,
        save_csv=True,
        save_path=p.join(args.base, "test_set_performance_ensembled.csv"),
    )

    dev_score = test_model_ensemble(
        model_1_location=args.base + "model_1_ft_hl/",
        model_2_location=args.base + "model_2_ft_hl/",
        model_type=args.model_type,
        X_test=X_valid,
        att_mask_test=att_mask_valid,
        y_test=y_valid,
        device=args.device_2,
        batch_size=args.batch_size,
        num_classes=num_classes,
        save_csv=True,
        save_path=p.join(args.base, "dev_set_performance_ensembled.csv"),
    )

    print(
        f"\nExperiment completed! Dev score: {dev_score:.4f}, Test score: {test_score:.4f}"
    )


if __name__ == "__main__":
    main()

# python script.py --base /path/to/experiment/ --seed 1234 --model_type RoBERTa --batch_size 32 --num_epochs 5 --patience 2
