#!/opt/conda/bin/python
# -*- coding: utf-8 -*-


from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from clinical_nlp.arguments import DataArguments, ModelArguments
from datasets import load_dataset
import evaluate
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import transformers
import datasets

datasets.disable_progress_bar()
import os
import sys

logger = logging.getLogger(__name__)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data = load_dataset(data_args.dataset_name, revision=data_args.dataset_revision)
    if model_args.task == "text-classification":
        metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(
                predictions=predictions, references=labels, average="weighted"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        le = LabelEncoder()
        le.fit(data["train"]["label"])
        data["train"] = data["train"].class_encode_column("label")
        data["validation"] = data["validation"].class_encode_column("label")
        data["test"] = data["test"].class_encode_column("label")
        label2id = dict(zip(le.classes_, range(len(le.classes_))))
        id2label = dict(zip(label2id.values(), label2id.keys()))
        num_lan = len(le.classes_)
        data["train"] = data["train"].align_labels_with_mapping(label2id, "label")
        data["validation"] = data["validation"].align_labels_with_mapping(
            label2id, "label"
        )
        data["test"] = data["test"].align_labels_with_mapping(label2id, "label")
        tokenized_train_ds = data["train"].map(preprocess_function, batched=True)
        tokenized_validation_ds = data["validation"].map(
            preprocess_function, batched=True
        )
        tokenized_test_ds = data["test"].map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        if model_args.peft:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    elif model_args.task == "ner":
        label_list = data["train"].features[f"ner_tags"].feature.names
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, add_prefix_space=True
        )

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(
                    batch_index=i
                )  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif (
                        word_idx != previous_word_idx
                    ):  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_data = data.map(tokenize_and_align_labels, batched=True)
        tokenized_train_ds = tokenized_data["train"]
        tokenized_validation_ds = tokenized_data["validation"]
        tokenized_test_ds = tokenized_data["test"]
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(
                predictions=true_predictions, references=true_labels
            )
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        label2id = dict(zip(label_list, range(len(label_list))))
        id2label = dict(zip(label2id.values(), label2id.keys()))
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )
        if model_args.peft:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds.select(
            range(int(len(tokenized_train_ds) * data_args.dataset_sample))
        ),
        eval_dataset=tokenized_validation_ds.select(
            range(int(len(tokenized_validation_ds) * data_args.dataset_sample))
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(f"{training_args.output_dir}/final")
