from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # path to the data
    train_file: Optional[str] = field(
        default='You/Path', metadata={"help": "A csv or a json file containing the training data."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": ("Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.")},
    )

    #--------------------------------------------------------Debug Args--------------------------------------------------#
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes or quicker training, truncate the number of training examples to this value if set.")},
    )
    #--------------------------------------------------------------------------------------------------------------------#

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # model_name_or_path contains config_name, tokenizer_name
    model_name_or_path: str = field(
        default='google-bert/bert-base-cased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # bce / ce / regression
    problem_type: str = field(
        default='single_label_classification',
        metadata={"help": "The type of problem to solve, e.g., 'single_label_classification', 'multi_label_classification', etc."}
    )
    # pretrained models downloaded to
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # fast tokenizer supported by a specific library
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # huggingface library token for auth
    token: str = field(
        default=None,
        metadata={"help": ("The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).")},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": ("Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.")},
    )

    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class TrainingArgumentDK(TrainingArguments):
    # batch_size each gpu
    per_device_train_batch_size: int = field(
        default=32,
    )
    # lr
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    # epochs
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    # label smoothing for CE
    label_smoothing_factor: float = field(
        default=.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing). In general, 0.1, If no smoothing, set 0"}
    )
    # mixed training
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    # save model every `save_steps` steps
    save_steps: int = field(
        default=500,
        metadata={
            "help": ("Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.")
        },
    )
    # model save path
    output_dir: str = field(
        default='/home/duke/nlpdk/test',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    # overwrite output dir
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": ("Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.")},
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    ddp_backend: Optional[str] = field(
        default='nccl',
        metadata={"help": "The backend to be used for distributed training", 
                  "choices": ["nccl", "gloo", "mpi", "ccl", "hccl", "cncl"],},
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={"help": ("When using distributed training, the value of the flag `find_unused_parameters` passed to `DistributedDataParallel`.")},
    )