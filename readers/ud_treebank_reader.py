from typing import List, Dict, Any, Optional
import pickle
from utils.word import Word
from readers.base import Reader
import pycountry
import config
from os import path


class UDTreebankReader(Reader):
    """
    Class for reading pre-processed UD treebanks.
    """
    @staticmethod
    def get_treebank_file(
            language_code: str, embedding: Optional[str] = "bert",
            test_file: Optional[bool] = False, valid_file: Optional[bool] = False) -> str:

        assert not (valid_file and test_file)

        lang = pycountry.languages.get(alpha_3=language_code)

        if embedding == "bert":
            embedding_file = "bert-base-multilingual-cased"
        elif embedding == "fasttext":
            embedding_file = f"cc.{lang.alpha_2}.300.bin"
        else:
            raise Exception("")

        if test_file:
            flag = "test"
        elif valid_file:
            flag = "dev"
        else:
            flag = "train"

        return path.join(
            config.DATA_ROOT,
            f"ud/ud-treebanks-v2.1/UD_{lang.name}/{lang.alpha_2}-um-{flag}-{embedding_file}.pkl"
        )

    @classmethod
    def read(cls, paths: List[str]) -> List[Word]:
        """
        Should be overriden with the logic to (i) read all words in the dataset and (ii) discover
        the values each unimorph attribute can take and place them in
        self._unimorph_attributes_to_values_dict.
        """
        raw_words: List[Dict[str, Any]] = []
        for path in paths:
            with open(path, "rb") as h:
                raw_words.extend(pickle.load(h))

        # Read all words and store them in self._words
        words = []
        for item in raw_words:
            words.append(Word(item["word"], item["embedding"].reshape(-1), 1, item["attributes"]))

        return words
