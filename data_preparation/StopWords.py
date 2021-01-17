from nltk.corpus import stopwords
from typing import List


stopWords = list(stopwords.words('english'))
stopWords.extend([".", ",", "!", "(", ")"])


def short_array(original: List[str], indexes: List[int]) -> List[str]:
    new_list = list()
    for index, value in enumerate(original):
        if index in indexes:
            continue
        new_list.append(value)
    return new_list


def removeStopWords(data):
    """Removes Stopwords from given data"""
    for i, (k, v) in enumerate(data.items()):
        remove_index = list()
        tokens = v.get('tokens')
        for index, token in enumerate(tokens):
            if token in list(stopWords):
                remove_index.append(index)
        if len(remove_index) == 0:
            continue
        data[k]["tokens"] = short_array(tokens, remove_index)
        for ind, (key, val) in enumerate(v.items()):
            if key.__eq__("tokens"):
                continue
            for label_type, rating in val.items():
                data[k][key][label_type] = short_array(rating, remove_index)
    return data
