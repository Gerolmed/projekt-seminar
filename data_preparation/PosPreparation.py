import copy
from typing import List, Tuple

from loading.LoadingUtils import LoadedData
from loading.Preprocessing import pos_tagger, isStopWord
from utils.Data import Data, PosData
from utils.DataProvider import DataProvider
from utils.DataSelector import DataSelector


class PosPreparation(DataProvider):

    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: DataSelector) -> Data:
        includePOStags = True
        rawData = copy.deepcopy(rawData)
        tagged_sentences: List[Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]] = list()
        pos_tags: List[Tuple[str, str]]

        stoppedTestLabels = dict()
        stopWordLabel = "O"

        x_train = []
        y_train: List[str] = []

        x_test = []
        y_test: List[str] = []

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")
            pos_tags = pos_tagger(tokens) if includePOStags else [("", "") for _ in tokens]

            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData[data_selector.type_name]  # todo add _uncertainty as well
                sentence = list()
                for index, token in enumerate(tokens):
                    sentence.append((token, data_selector.type_symbol if labels[index].endswith(
                        data_selector.type_symbol) else "O"))
                tagged_sentences.append((dataKey, sentence, pos_tags))

        for group, tagged, pos_tags in tagged_sentences:
            for index in range(len(tagged)):

                token = extract_word(tagged)[index]
                features = extract_features(extract_word(tagged), extract_label(tagged), index, pos_tags)
                label = tagged[index][1]

                if group in test_ids:
                    x_test.append(features)
                    y_test.append(label)
                    if isStopWord(token):
                        stoppedTestLabels.setdefault(len(y_test)-1, stopWordLabel)
                else:
                    x_train.append(features)
                    y_train.append(label)

        return PosData(x_train, y_train, x_test, y_test, stoppedTestLabels)


def extract_features(token, labels, index, pos_tags):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': token[index],
        'prev_label': '' if index == 0 else labels[index - 1],
        'next_label': '' if index == len(token) - 1 else labels[index + 1],
        'is_numeric': token[index].isdigit(),
        'length': len(token[index]),
        'POS_tag': pos_tags[index][1]
    }


def extract_word(tagged_sentence):
    return [w for w, t in tagged_sentence]


def extract_label(tagged_sentence):
    return [t for w, t in tagged_sentence]
