from dataclasses import dataclass, field
from typing import Optional
import os
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
    train_data: Optional[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )

    sub_data_name: Optional[str] = field(
        default=None, metadata={"help": "Path to sub data"}
    )

    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: Optional[str]= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    #def __post_init__(self):
    #    if not os.path.exists(self.train_data):
    #        raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    # is negatice cross device 
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices, equal to enlarge batch size in a more memory-saved way"})

    # temperature: used in logits/t
    temperature: Optional[float] = field(default=0.02)

    # freeze PE
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})

    # pooling method
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})

    # is L2 normlized
    normlized: bool = field(default=True)

    # use negatives of A as B's negatices
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})

    # batch_size each gpu
    per_device_train_batch_size: int = field(
        default=32,
    )

    # lr
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})

    # epochs
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embedding_output'),
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