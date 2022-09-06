import datasets
import mlflow
import numpy as np
import os
from persuasion4good.common import Task
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy,
    DataCollatorForLanguageModeling
)

class FinetuningTask(Task):
    def __init__(
        self,
        model = "microsoft/DialoGPT-small",
        tokenizer = "microsoft/DialoGPT-small",
        config = "microsoft/DialoGPT-small"
    ):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            return_special_tokens_mask = True
        )
        self._config = AutoConfig.from_pretrained(config)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            config = config
        )
        self._dataset = self._load_dataset()
        self._setup_mlflow()

    def _load_dataset(
        self,
        keep_in_memory = True
    ):

        dataset_source_dir = self.conf["dataset_source_dir"]
        dataset = datasets.load_from_disk(
            dataset_source_dir,
            keep_in_memory = keep_in_memory
        )
        
        return dataset

    def _filter(
        self,
        min_utterance_len: int = 2
    ):
        self._dataset = self._dataset.filter(
            lambda example: len(example['label']) > min_utterance_len \
                and len(example['context']) > min_utterance_len
        )

    def get_max_encoded_length(
        self,
        sample_size = 100,
        embedding_size_quantile: float = 0.95
    ):
        embedded_lengths = []

        for _ in range(sample_size):
            embedded = self._tokenizer(
                self._dataset["train"].shuffle()[0]["context"],
                return_tensors = "pt",
                return_attention_mask = False
            )
            embedded_lengths.append(
                len(torch.flatten(embedded["input_ids"]))
            )

        quantile = np.quantile(embedded_lengths, embedding_size_quantile)
        return int(quantile)
  
    def tokenize(
        self,
        batch,
        feature,
        eos: bool = True,
        max_length: int = 1024,
        pad_to_multiple_of: int = 8,
        eos_string: str = "<EOS>"
    ):
        self._tokenizer.pad_token = self._tokenizer.eos_token
  
        if eos:
            batch[feature] = batch[feature].replace(
                eos_string,
                self._tokenizer.eos_token
            )
        
        # Round max_length to the nearest multiple of pad_to_multiple_of
        remainder = pad_to_multiple_of - (max_length % pad_to_multiple_of)
        max_length_multiple = max_length + remainder
        
        input_ids = self._tokenizer(
            batch[feature],
            return_tensors = "pt",
            return_attention_mask = False,
            padding = "max_length",
            truncation = "longest_first",
            max_length = max_length_multiple,
            pad_to_multiple_of = pad_to_multiple_of
        )["input_ids"][0]
        
        result_key = "labels"
        if feature != "label":
            result_key = "input_ids"
            
        result = {f"{result_key}": input_ids}
        return result

    def _prepare_dataset(
        self,
        sample_size: int = 100,
        embedding_size_quantile: float = 0.95
    ):
        max_embedding_length = self.get_max_encoded_length(
            sample_size = sample_size,
            embedding_size_quantile = embedding_size_quantile
        )

        self.logger.info("Tokenizing with padding to ", max_input_length)
        self._dataset = self._dataset.map(
            lambda x: self.tokenize(
                x,
                feature = "context",
                max_length = max_embedding_length
            ),
            remove_columns = ["context"]
        )
        self._dataset = self._dataset.map(
            lambda x: self.tokenize(
                x,
                feature = "label",
                eos = False,
                max_length = max_embedding_length
            ),
            remove_columns = ["label"]
        )

    def _setup_mlflow(self):

        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.conf["mlflow_experiment_name"]
        os.environ["MLFLOW_FLATTEN_PARAMS"] = int(self.conf["mlflow_flatten_params"])
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = int(self.conf["mlflow_log_artifacts"])
        os.environ["MLFLOW_NESTED_RUN"] = int(self.conf["mlflow_nested_run"])

    def _compute_metrics(eval_loss: float):
    
        loss = torch.exp(eval_loss).float()
        return loss

    def get_trainer(
        self
    ):

        self._tokenizer.pad_token = self._tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(
            tokenizer = self._tokenizer,
            mlm = False,
            pad_to_multiple_of = int(self.conf["pad_to_multiple_of"])
        )

        args = TrainingArguments(
            output_dir = self.conf["trainer_output_dir"],
            evaluation_strategy = IntervalStrategy.STEPS, # "steps"
            eval_steps = 50, # Evaluation and Save happens every 50 steps
            save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
            per_device_train_batch_size = 4,
            per_device_eval_batch_size = 4,
            eval_accumulation_steps = 1,
            learning_rate = 5e-5,
            weight_decay = 0.0,
            adam_epsilon = 1e-8,
            warmup_steps = 0.0,
            max_grad_norm = 1.0,
            num_train_epochs = 10000.0,
            logging_steps = 1000,
            no_cuda = False,
            overwrite_output_dir = True,
            seed = 42,
            local_rank = -1,
            fp16 = False,
            metric_for_best_model = 'eval_loss',
            load_best_model_at_end = True,
            disable_tqdm = False,
            prediction_loss_only=True
        )

        trainer = Trainer(
            data_collator = collator,
            compute_metrics = self.compute_metrics,
            model = self._model,
            args = args,
            train_dataset = self._dataset["train"],
            eval_dataset = self._dataset["test"],
            tokenizer = self._tokenizer,
            callbacks = [EarlyStoppingCallback(
                early_stopping_patience = 5,
                early_stopping_threshold = 0.0005
            )]
        )

        self._trainer = trainer

    def train(self, artifact_path = "model"):

        torch.cuda.empty_cache()
        model_info = None
        with mlflow.start_run(nested = True) as run:
            self._trainer.train()
            metrics = self._trainer.evaluate()
            self.logger.info(f"Metrics: {metrics}")
            mlflow.log_metrics(metrics)
            model = self._trainer.model
            model_info = mlflow.pytorch.log_model(model, artifact_path = artifact_path)

        return model_info