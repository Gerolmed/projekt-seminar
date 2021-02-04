import copy
from typing import List, Tuple

from loading.LoadingUtils import LoadedData
from utils.Data import Data, PosData
from utils.DataProvider import DataProvider
from utils.DataSelector import DataSelector


class PosPreparation(DataProvider):

    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: DataSelector) -> Data:
        rawData = copy.deepcopy(rawData)
        tagged_sentences: List[Tuple[str, List[Tuple[str, str]]]] = list()

        x_train = []
        y_train: List[str] = []

        x_test = []
        y_test: List[str] = []

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData[data_selector.type_name]  # todo add _uncertainty as well
                sentence = list()
                for index, token in enumerate(tokens):
                    sentence.append((token, data_selector.type_symbol if labels[index].endswith(data_selector.type_symbol) else "O"))
                tagged_sentences.append((dataKey, sentence))

        for group in tagged_sentences:
            tagged = group[1]
            for index in range(len(tagged)):

                features = extract_features(extract_word(tagged), index)
                label = tagged[index][1]

                if group[0] in test_ids:
                    x_test.append(features)
                    y_test.append(label)
                else:
                    x_train.append(features)
                    y_train.append(label)

        return PosData(x_train, y_train, x_test, y_test)


def extract_features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def extract_word(tagged_sentence):
    return [w for w, t in tagged_sentence]
