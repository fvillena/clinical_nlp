from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_revision: Optional[str] = field(
        default=None, metadata={"help": "The name of the revision of the dataset."}
    )
    dataset_sample: Optional[float] = field(
        default=1.0, metadata={"help": "The percentage of the dataset to use."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    api_key: Optional[str] = field(
        default=None, metadata={"help": "The API key to use for the model."}
    )
    few_shot: bool = field(
        default=False, metadata={"help": "If true, will use the few-shot model."}
    )
    openai_model_name: Optional[str] = field(
        default="gpt-3.5-turbo", metadata={"help": "The name of the openai model to use."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    peft: bool = field(
        default=False,
        metadata={"help": "Whether to use the PEFT."},
    )
    task: str = field(
        default="text-classification",
        metadata={"help": "The task to use the model for."},
    )

@dataclass
class PipelineArguments:
    batch_size: int = field(
        default=64, metadata={"help": "The batch size to use for the pipeline."}
    )
    device: int = field(
        default=0, metadata={"help": "The device to use for the pipeline."}
    )
    truncation: bool = field(
        default=True, metadata={"help": "Whether to truncate the input."}
    )

@dataclass
class PredictionArguments:
    prediction_file: str = field(
        metadata={"help": "The file to write predictions to."}
    )