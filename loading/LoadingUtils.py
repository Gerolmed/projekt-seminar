import random
from typing import Dict, Union, List
import json
from data_preparation.StopWords import removeStopWords
from utils.Data import Data, BasicData

Tokens = List[str]
Review = Dict[str, Union[Dict[str, List[str]], Tokens]]
LoadedData = Dict[str, Review]
RawData = Dict[str, Dict[str, Union[Dict[str, List[str]], List[str]]]]
seed: int = 1234567890


class LoadingUtils:
    @staticmethod
    def read_data(filename: str) -> [Data, LoadedData]:

        extraction_of = 'sentiments'

        rawData: RawData = LoadingUtils.__open_file(filename)

        # possible preprocessing: lowercasing of tokens
        for i, (k, v) in enumerate(rawData.items()):
            tokens = v.get('tokens')
            tokens = [token.lower() for token in tokens]
            rawData[k]['tokens'] = tokens

        # remove Stopwords
        rawData: RawData = removeStopWords(rawData)

        keys = list(rawData.keys())
        random.seed(seed)
        random.shuffle(keys)  # Shuffle keys

        split_parameter = round(len(keys) * 0.7)
        keys_train = keys[:split_parameter]
        keys_test = keys[split_parameter:]

        train_tokens = [rawData[k]['tokens'] for k in keys_train]
        train_labels = list()
        train_labels_uncertainty = list()
        for k in keys_train:
            curr_users = [s for s in rawData[k].keys() if s != 'tokens']
            # for illlustration only the annotation of one user is used here -> curr_users[0]
            train_labels.append(rawData[k][curr_users[0]][extraction_of])
            train_labels_uncertainty.append(rawData[k][curr_users[0]][extraction_of + '_uncertainty'])
        test_tokens = [rawData[k]['tokens'] for k in keys_test]
        test_labels = list()
        test_labels_uncertainty = list()
        for k in keys_test:
            curr_users = [s for s in rawData[k].keys() if s != 'tokens']
            # for illustration only the annotation of one user is used here -> curr_users[0]
            test_labels.append(rawData[k][curr_users[0]][extraction_of])
            test_labels_uncertainty.append(rawData[k][curr_users[0]][extraction_of + '_uncertainty'])

        all_labelclasses = set()
        for ds in [train_labels, test_labels]:
            for row in ds:
                all_labelclasses.update(row)
        all_labelclasses = list(all_labelclasses)
        all_labelclasses.sort()

        labelclass_to_id = dict(zip(all_labelclasses, list(range(len(all_labelclasses)))))

        n_tags = len(list(labelclass_to_id.keys()))
        return [BasicData(train_tokens, train_labels, test_tokens, test_labels, n_tags, labelclass_to_id), rawData]

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
