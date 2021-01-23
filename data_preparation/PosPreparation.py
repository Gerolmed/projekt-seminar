import random
from typing import List, Tuple

from loading.LoadingUtils import LoadedData, seed
from utils.Data import Data, PosData
from utils.DataProvider import DataProvider


class PosPreparation(DataProvider):

    def execute(self, data: Data, rawData: LoadedData) -> Data:

        tagged_sentences: List[List[Tuple[str, str]]] = list()

        x_all = []
        y_all: List[str] = []

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData["sentiments"]  # todo add _uncertainty as well
                sentence = list()
                for index, token in enumerate(tokens):
                    sentence.append((token, labels[index]))
                tagged_sentences.append(sentence)

        for tagged in tagged_sentences:
            for index in range(len(tagged)):
                x_all.append(features(extract_word(tagged), index))
                y_all.append(tagged[index][1])

        random.seed(seed)
        random.shuffle(x_all)

        random.seed(seed)
        random.shuffle(y_all)

        split_parameter = round(len(y_all) * 0.7)

        return PosData(x_all[split_parameter:], y_all[split_parameter:], x_all[:split_parameter], y_all[:split_parameter])


def features(sentence, index):
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
