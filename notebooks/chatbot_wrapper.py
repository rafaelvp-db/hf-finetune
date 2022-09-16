import torch
import numpy as np
import mlflow.pyfunc

from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatbotWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to use HuggingFace Chatbot / DialoGPT Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """

        self._tokenizer = AutoTokenizer.from_pretrained(context.artifacts["hf_tokenizer_path"])
        self._model = AutoModelForCausalLM.from_pretrained(context.artifacts["hf_model_path"])
        
    def ask_question(
        self,
        question,
        chat_history_ids = [],
        max_new_tokens = 25,
        temperature = 1.0,
        repetition_penalty = 10.0,
        do_sample = True,
        top_k = 5,
        top_p = 0.9,
        no_repeat_ngram_size = 3,
        length_penalty = 20.0
        ):
        """
            This function is a wrapper on top of our fine tuned model.
            It preprocesses and tokenizes our inputs, and calls the generate function
            from the DialoGPT model we trained.
        """
        
        new_user_input_ids = self._tokenizer.encode(
            str(question) + str(self._tokenizer.eos_token),
            return_tensors = 'pt'
        )

        chat_history_tensor = torch.tensor(chat_history_ids)
        if (len(chat_history_ids) > 0):
            # Beginning of conversation
            bot_input_ids = torch.cat(
                [
                    chat_history_tensor,
                    torch.flatten(new_user_input_ids)
                ], 
                dim = -1
            )
            bot_input_ids = torch.reshape(
                bot_input_ids,
                shape = (1, -1)
            )
        else:
            # Continuing an existing conversation
            bot_input_ids = new_user_input_ids
            
        chat_history_ids = self._model.generate(
            bot_input_ids,
            eos_token_id = self._tokenizer.eos_token_id,
            max_new_tokens = max_new_tokens,
            pad_token_id = self._tokenizer.eos_token_id,  
            no_repeat_ngram_size = no_repeat_ngram_size,
            do_sample  = do_sample, 
            top_k = top_k, 
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            temperature = temperature,
            length_penalty = length_penalty,
            forced_eos_token_id = self._tokenizer.eos_token_id
        )
        
        print(chat_history_ids)

        # Using our tokenizer to decode the outputs from our model
        # (e.g. convert model outputs to text)
        
        answer = self._tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens = True
        )

        return answer, chat_history_ids

    def predict(self, context, model_input):

        answer, chat_history_ids = self.ask_question(
            question = model_input["question"],
            chat_history_ids = model_input["chat_history_ids"]
        )
        
        result = {
            "answer": answer,
            "chat_history_ids": chat_history_ids[0].tolist()
        }

        return result


def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return ChatbotWrapper(data_path)