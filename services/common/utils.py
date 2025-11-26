from services.common.common import logger, processControl, log_
import json

import time
import os
from os.path import isdir
import spacy
import unicodedata
import re

class cleanNLPString:
    def __init__(self, lang: str):
        self.lang = lang
        self.model = "en_core_web_sm"
        if self.lang == 'es':
            self.model = "es_core_news_sm"

        # Load the SpaCy model
        try:
            self.nlp = spacy.load(self.model)
        except Exception as e:
            raise ValueError(f"Failed to load SpaCy model '{self.model}': {e}")

    def sentences(self, text: str):
        """Split text into sentences using SpaCy."""
        doc = self.nlp(text)
        return list(doc.sents)

    def clean(self, text: str, minLengthText=70) -> str:

        """Clean and preprocess text."""
        # Normalize and remove non-ASCII characters
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'@\w+', r'@\g<0>', text)  # Conserva menciones
        text = re.sub(r'[^\w\s@áéíóúÁÉÍÓÚñÑ]', '', text)

        # Validate input length
        if not isinstance(text, str) or len(text) < minLengthText:
            return None  # Discard sentences below the length threshold

        # Process the text with SpaCy
        doc = self.nlp(text)

        # Cleaning and tokenization
        cleaned_tokens = []
        for token in doc:
            if not token.like_num:  # Exclude numbers but retain stopwords and punctuation
                cleaned_tokens.append(token.text.lower())

        # Join the cleaned tokens into a single string
        cleaned_text = ' '.join(cleaned_tokens).strip()

        # Return None if text ends up empty
        return cleaned_text if cleaned_text else None


def mkdir(dir_path):
    """
    @Desc: Creates directory if it doesn't exist.
    @Usage: Ensures a directory exists before proceeding with file operations.
    """
    if not isdir(dir_path):
        os.makedirs(dir_path)


def dbTimestamp():
    """
    @Desc: Generates a timestamp formatted as "YYYYMMDDHHMMSS".
    @Result: Formatted timestamp string.
    """
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp

class configLoader:
    """
    @Desc: Loads and provides access to JSON configuration data.
    @Usage: Instantiates with path to config JSON file.
    """
    def __init__(self, config_path='config.json'):
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        with open(realConfigPath, 'r') as config_file:
            return json.load(config_file)

    def get_environment(self):
        environment =  self.config.get("environment", None)
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        return self.config.get("defaults", {})

    def get_models(self):
        return self.config.get("models", {})

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out



