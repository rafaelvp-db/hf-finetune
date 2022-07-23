# Databricks notebook source
from transformers import AutoTokenizer, PreTrainedTokenizer

class ConversationDataset(Dataset):
    def __init__(
      self,
      tokenizer: "microsoft/DialoGPT2",
      model_type = "gpt2",
      args,
      df,
      block_size=512
   ):

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        block_size = block_size - (self._tokenizer.model_max_length - self._tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        pathlib.Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = self._construct_conv(row, tokenizer)
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    # create dataset suitable for our model
    def _construct_conv(self, row, tokenizer, eos = True):
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = list(reversed([self._tokenizer.encode(x) + [self._tokenizer.eos_token_id] for x in row]))
        conv = flatten(conv)
        return conv

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
