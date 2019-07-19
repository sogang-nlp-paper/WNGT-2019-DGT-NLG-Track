from pytorch_transformers import BertConfig, BertTokenizer
from torchtext.vocab import Vocab

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


class BertBasedVocab(Vocab):

    def __init__(self, counter, specials=['[UNK]'], bert_model_name='bert-base-cased'):
        self.freqs = counter
        counter = counter.copy()

        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.stoi = {k: self.tokenizer.convert_tokens_to_ids(k) for k in (list(counter) + specials)}
        self.set_itos()

    def set_itos(self):
        self.itos = [None] * self.bert_config.vocab_size
        for w in list(self.stoi.keys()):
            idx = self.stoi[w]
            self.itos[idx] = w
