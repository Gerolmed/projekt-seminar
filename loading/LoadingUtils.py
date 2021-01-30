from typing import Dict, Union, List
import json
from loading.StopWords import removeStopWords
from utils.Data import Data, BasicData

Tokens = List[str]
Review = Dict[str, Union[Dict[str, List[str]], Tokens]]
LoadedData = Dict[str, Review]
RawData = Dict[str, Dict[str, Union[Dict[str, List[str]], List[str]]]]
seed: int = 1234567890


def merge_sentiments(sentiments: List[str], sentiments_uncertainty: List[str]):
    merged: List[str] = []

    for index, sentiment in enumerate(sentiments):
        if sentiment != "O":
            merged.append(sentiment)
        else:
            merged.append(sentiments_uncertainty[index])

    return merged


class LoadingUtils:
    @staticmethod
    def read_data(filename: str, testIdsFile: str) -> [Data, LoadedData]:

        raw_data: RawData = LoadingUtils.__open_file(filename)
        test_ids = LoadingUtils.__open_test_ids(testIdsFile)

        # lowercasing of tokens
        for i, (k, v) in enumerate(raw_data.items()):
            tokens = v.get('tokens')
            tokens = [token.lower() for token in tokens]
            raw_data[k]['tokens'] = tokens

        # remove Stopwords
        raw_data: RawData = removeStopWords(raw_data, test_ids)

        # currently just merge uncertain in sentiments
        for (sentence_key, value) in raw_data.items():
            for review_key, review_data in value.items():
                if review_key == "tokens":
                    continue
                raw_data[sentence_key][review_key]["sentiments"] = merge_sentiments(review_data.get("sentiments"),
                                                                                    review_data.get(
                                                                                        "sentiments_uncertainty"))
        return [raw_data, test_ids]

    @staticmethod
    def __open_file(filename: str):
        with open(file=filename, mode='r', encoding='utf8') as infile:
            data = json.load(infile)
            data_new: LoadedData = dict()
            for review_id, review_data in list(data.items()):
                v_new = data_new.setdefault(review_id, dict())
                for field, field_data in list(review_data.items()):
                    v1_new = v_new.setdefault(field, dict())
                    if field != 'tokens':
                        for k2, v2 in list(field_data.items()):
                            v2_new = [l.replace('B_', '').replace('I_', '') for l in v2]
                            v1_new[k2] = v2_new
                    else:
                        v_new[field] = field_data
        return data_new

    @staticmethod
    def __open_test_ids(filename: str):
        ids: List[str] = []

        with open(file=filename, mode='r', encoding='utf8') as infile:
            data = json.load(infile)
            ids = list(data.get("test_IDs"))
        return ids
