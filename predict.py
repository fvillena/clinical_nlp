#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline, HfArgumentParser
from datasets import load_dataset
from clinical_nlp.arguments import DataArguments, ModelArguments, PipelineArguments, PredictionArguments
from clinical_nlp.models import IclModel

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, PipelineArguments, PredictionArguments))
    data_args, model_args, pipeline_arguments, prediction_arguments = parser.parse_args_into_dataclasses()
    data = load_dataset(data_args.dataset_name, revision=data_args.dataset_revision)
    test_data = data["test"]
    if data_args.dataset_sample:
        test_data = test_data.select(range(int(len(test_data)*data_args.dataset_sample)))
    if "http" in model_args.model_name_or_path:
        classes = list(set(test_data["label"]))
        model = IclModel(model_args.model_name_or_path)
        model.contextualize(
            system_message="Eres un asistente serio que sólo da respuestas precisas y concisas que recibirá diagnósticos en Español y deberás sólo responder con el nombre de la especialidad en Español a la cual debe enviarse el diagnóstico. Las especialidades disponibles son: <classes>.",
            user_template="¿A qué especialidad debo enviar el diagnóstico \"<x>\"?.",
            classes=classes
        )
        test_data = test_data.map(lambda x: {"prediction": model.predict(x["text"])}, num_proc=12)
        true = test_data['label']
        predicted = test_data['prediction']
    else:
        pipe = pipeline(
            "text-classification", 
            model=model_args.model_name_or_path,
            tokenizer=model_args.tokenizer_name,
            device=pipeline_arguments.device
        )
        predictions = pipe(
            test_data['text'],
            batch_size=pipeline_arguments.batch_size,
            truncation=pipeline_arguments.truncation
        )
        true = test_data['label']
        predicted = [p['label'] for p in predictions]
    with open(prediction_arguments.prediction_file, 'w', encoding="utf-8") as f:
        for t,p in (zip(true,predicted)):
            f.write(f'{t}\t{p}\n')