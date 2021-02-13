from nltk import PorterStemmer, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from typing import Dict, Union, List, Tuple
import pandas as pd

RawData = Dict[str, Dict[str, Union[Dict[str, List[str]], List[str]]]]

stopWords = list(stopwords.words('english'))
stopWords.extend([".", ",", "!", "(", ")", '"', "-", "'", ":", ";", "?", "=", "<", ">", "https", "div", "&", "/", "*",
                  "[", "]"])

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def short_array(original: List[str], indexes: List[int]) -> List[str]:
    new_list = list()
    for index, value in enumerate(original):
        if index in indexes:
            continue
        new_list.append(value)
    return new_list


def removeStopWords(data: RawData, test_data_ids: List[str]):
    """Removes Stopwords from given data"""
    for i, (k, v) in enumerate(data.items()):
        if k in test_data_ids:
            continue
        remove_index = list()
        tokens = v.get('tokens')
        for index, token in enumerate(tokens):
            if isStopWord(token):
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


def isStopWord(token):
    return token in list(stopWords)


def stemming(data: RawData):
    for i, (k, v) in enumerate(data.items()):
        tokens = v.get('tokens')
        tokens = [ps.stem(token) for token in tokens]
        data[k]['tokens'] = tokens
    return data


def lemmantising(data: RawData):
    for i, (k, v) in enumerate(data.items()):
        tokens = v.get('tokens')
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        data[k]['tokens'] = tokens
    return data


def dataToDataFrame(data: RawData, test_data_ids: List[str]) -> pd.DataFrame:
    dataFrame = pd.DataFrame(
        columns=["SentenceNr", "inTestIDs", "Token", "POS_tag", "a_tag", "a_unc", "s_tag", "s_unc", "s_diff",
                 "m_tag", "m_unc"])
    sentenceCounter: int = 0
    for dataKey, dataValue in data.items():
        tokens: List[str] = dataValue.get("tokens")
        for i, token in enumerate(tokens):
            if token == "'":
                tokens[i] = '"'
        pos_tags = pos_tagger(tokens)
        inTestDataIds: bool = True if dataKey in test_data_ids else False
        for reviewKey, dataData in dataValue.items():
            if reviewKey == "tokens":
                continue

            for index, token in enumerate(tokens):
                pos_tag: str = pos_tags[index][1]
                a_tag: str = dataData.get("aspects")[index]
                a_unc: str = dataData.get("aspects_uncertainty")[index]
                s_tag: str = dataData.get("sentiments")[index]
                s_unc: str = dataData.get("sentiments_uncertainty")[index]
                s_diff: str = dataData.get("sentiments_difficulty")[index]
                m_tag: str = dataData.get("modifiers")[index]
                m_unc: str = dataData.get("modifiers_uncertainty")[index]

                newRow = {"SentenceNr": sentenceCounter, "inTestIDs": inTestDataIds, "Token": token, "POS_tag": pos_tag,
                          "a_tag": a_tag,
                          "a_unc": a_unc, "s_tag": s_tag, "s_unc": s_unc, "s_diff": s_diff,
                          "m_tag": m_tag, "m_unc": m_unc}

                dataFrame = dataFrame.append(newRow, ignore_index=True)
            sentenceCounter += 1

    return dataFrame


def dataFrameToCSV(dataFrame: pd.DataFrame):
    dataFrame.to_csv('Data.csv', index=True)


def pos_tagger(tokens: List[str]) -> List[Tuple[str, str]]:
    taggedTokens: List[Tuple[str, str]] = pos_tag(tokens)
    return taggedTokens


# The following method was only implemented for adding TestIDs to an existing csv file
"""
def addTestIDs(data: RawData, test_ids: List[str], file_name: str):
    df = pd.read_csv(file_name, index_col=0)
    new_col = []
    for dataKey, dataValue in data.items():
        tokens: List[str] = dataValue.get("tokens")
        for reviewKey, dataData in dataValue.items():
            if reviewKey == "tokens":
                continue
            for token in tokens:
                new_col.append(True if dataKey in test_ids else False)
    df.insert(loc=2, column="inTestIDs", value=new_col)
    df.to_csv("Data.csv", index=False)
"""
