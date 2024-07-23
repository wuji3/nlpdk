import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    sub_data_name: Optional[str] = field(
        default=None, metadata={"help": "Path to sub data"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    # def __post_init__(self):
    #     if not os.path.exists(self.train_data):
    #         raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")



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
    # logging steps
    logging_steps: float = field(
        default=500,
        metadata={
            "help": ("Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.")
        },
    )
    # model save path
    output_dir: str = field(
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rerank_output'),
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    # overwrite output dir
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": ("Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.")},
    )
    # gradient accumulation used only if batch size is small
    gradient_accumulation_steps: int = field(
            default=1,
            metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
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
