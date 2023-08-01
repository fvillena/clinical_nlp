#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline, HfArgumentParser
from datasets import load_dataset
from clinical_nlp.arguments import (
    DataArguments,
    ModelArguments,
    PipelineArguments,
    PredictionArguments,
)
from clinical_nlp.models import IclModel

if __name__ == "__main__":
    parser = HfArgumentParser(
        (DataArguments, ModelArguments, PipelineArguments, PredictionArguments)
    )
    (
        data_args,
        model_args,
        pipeline_arguments,
        prediction_arguments,
    ) = parser.parse_args_into_dataclasses()
    data = load_dataset(data_args.dataset_name, revision=data_args.dataset_revision)
    test_data = data["test"]
    if data_args.dataset_sample:
        test_data = test_data.select(
            range(int(len(test_data) * data_args.dataset_sample))
        )
    if model_args.task == "text-classification":
        if "http" in model_args.model_name_or_path:
            classes = list(set(test_data["label"]))
            model = IclModel(model_args.model_name_or_path)
            model.contextualize(
                system_message="Eres un asistente serio que sólo da respuestas precisas y concisas que recibirá diagnósticos en Español y deberás sólo responder con el nombre de la especialidad en Español a la cual debe enviarse el diagnóstico. Las especialidades disponibles son: <classes>.",
                user_template='¿A qué especialidad debo enviar el diagnóstico "<x>"?.',
                classes=classes,
                retry_message="No encuentro en tu mensaje ninguna de las especialidades de la lista de especialidades disponibles. Por favor, intenta nuevamente."
            )
            test_data = test_data.map(
                lambda x: {"prediction": model.predict(x["text"])}, num_proc=12
            )
            true = test_data["label"]
            predicted = test_data["prediction"]
        else:
            pipe = pipeline(
                "text-classification",
                model=model_args.model_name_or_path,
                tokenizer=model_args.tokenizer_name,
                device=pipeline_arguments.device,
            )
            predictions = pipe(
                test_data["text"],
                batch_size=pipeline_arguments.batch_size,
                truncation=pipeline_arguments.truncation,
            )
            true = test_data["label"]
            predicted = [p["label"] for p in predictions]
        with open(prediction_arguments.prediction_file, "w", encoding="utf-8") as f:
            for t, p in zip(true, predicted):
                f.write(f"{t}\t{p}\n")
    elif model_args.task == "ner":
        from transformers import (
            AutoTokenizer,
            TokenClassificationPipeline,
            AutoModelForTokenClassification,
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
        def process_true_labels(example):
            true = [-100 if i == -100 else pipe.model.config.id2label[i] for i in example]
            interim = []
            for i, piece in enumerate(true):
                if i == 0:
                    interim.append("O")
                elif i == len(true)-1:
                    interim.append("O")
                elif piece != -100:
                    interim.append(piece)
                else:
                    last_piece = interim[i-1]
                    interim.append(last_piece if not last_piece.startswith("B-") else last_piece.replace("B-", "I-"))
            true = interim
            return true
        def process_predicted_labels(prediction,sentence):
            interim = ["O"]*len(sentence)
            for result in prediction:
                interim[result["index"]] = result["entity"]
            predicted = interim
            return predicted
        class MyTokenClassificationPipeline(TokenClassificationPipeline):
            def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
                tokenizer_params = preprocess_params.pop("tokenizer_params", {})
                truncation = (
                    True
                    if self.tokenizer.model_max_length
                    and self.tokenizer.model_max_length > 0
                    else False
                )
                inputs = self.tokenizer(
                    sentence,
                    return_tensors=self.framework,
                    truncation=truncation,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                    is_split_into_words=True,
                    **tokenizer_params,
                )
                inputs.pop("overflow_to_sample_mapping", None)
                num_chunks = len(inputs["input_ids"])

                for i in range(num_chunks):
                    if self.framework == "tf":
                        raise Exception(
                            "TensorFlow pipelines are currently not supported."
                        )
                    else:
                        model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
                    if offset_mapping is not None:
                        model_inputs["offset_mapping"] = offset_mapping
                    model_inputs["sentence"] = sentence if i == 0 else None
                    model_inputs["is_last"] = i == num_chunks - 1

                    yield model_inputs

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, add_prefix_space=True
        )
        tokenized_test_data = test_data.map(tokenize_and_align_labels, batched=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path
        )
        pipe = MyTokenClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0,
            aggregation_strategy="none",
        )
        sentences = list(map(pipe.tokenizer.convert_ids_to_tokens, tokenized_test_data["input_ids"]))
        true = list(map(process_true_labels, tokenized_test_data["labels"]))
        predictions = pipe(test_data["tokens"])
        predicted = []
        for prediction, sentence in zip(predictions, sentences):
            predicted.append(process_predicted_labels(prediction,sentence))
        with open(prediction_arguments.prediction_file, "w", encoding="utf-8") as f:
            for i in range(len(sentences)):
                for piece, t, p in zip(sentences[i], true[i], predicted[i]):
                    f.write(f"{piece}\t{t}\t{p}\n")
                f.write("\n")

