# Load model directly
import enum
from abc import ABC, abstractmethod

from deep_translator import GoogleTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslatorType(enum.Enum):
    google = 0
    helsinki_nlp_ops = 1


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, query: str) -> str:
        pass


class GoogleTranslatorCls(BaseTranslator):
    def __init__(self):
        self.translator = GoogleTranslator(source="ru", target="en")

    def translate(self, query: str) -> str:
        return self.translator.translate(text=query)


class Helsinki_NLP_otus(BaseTranslator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

    def translate(self, query: str) -> str:
        input_ids = tokenizer.encode(query, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded


class Translator(BaseTranslator):
    def __init__(self, translator_type=TranslatorType.google):
        self.translator_type = translator_type

        if translator_type == TranslatorType.google:
            self.translator = GoogleTranslatorCls()
        elif translator_type == TranslatorType.helsinki_nlp_ops:
            pass

    def translate(self, query: str) -> str:
        return self.translator.translate(query)
