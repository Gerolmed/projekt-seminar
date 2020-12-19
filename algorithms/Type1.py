

from utils.Algorithm import Algorithm
from utils.Data import Data
from utils.Result import Result
import numpy as np
import pandas as pd


class Type1(Algorithm):

    def get_name(self) -> str:
        return "type_1"

    def execute(self, data: Data) -> Result:
        # start classification

        # train classifier -> generate word lists per labelclass
        # count frequencies of labelclasses per term
        term_to_labelclass_to_freq = dict()
        for idx, tokens in enumerate(data.train_tokens):
            labels = data.train_labels[idx]
            for t, l in zip(tokens, labels):
                if l != 'O':
                    labelclass_to_freq = term_to_labelclass_to_freq.setdefault(t, dict())
                    freq = labelclass_to_freq.setdefault(l, 0)
                    labelclass_to_freq[l] = freq + 1

        # get most frequent labelclasses per term
        term_to_most_freq_labelclass = dict()
        for term, labelclass_to_freq in term_to_labelclass_to_freq.items():
            index_argmax = np.argmax(labelclass_to_freq.values())
            most_freq_labelclass = list(labelclass_to_freq.keys())[index_argmax]
            term_to_most_freq_labelclass[term] = most_freq_labelclass

        # -> term_to_most_freq_labelclass is the "mapping" which is used as classifier in this simple exammple
        # train phase is over

        ## evaluation (test classifier)
        # predict labels of test data (important: use only data.test_tokens, and do not use data.test_labels at all!!)
        data.test_labels_pred = list()
        for tokens in data.test_tokens:
            labels_pred = list()
            for tok in tokens:
                # classify token with mapping by term_to_most_freq_labelclass
                if tok in term_to_most_freq_labelclass.keys():
                    labels_pred.append(term_to_most_freq_labelclass[tok])
                else:
                    labels_pred.append('O')
            data.test_labels_pred.append(labels_pred)
        # compute confusion matrix
        conf_matrix = np.zeros((data.n_tags, data.n_tags))
        for i, tokens in enumerate(data.test_tokens):
            for j, _ in enumerate(tokens):
                class_id_true = data.labelclass_to_id[data.test_labels[i][j]]
                class_id_pred = data.labelclass_to_id[data.test_labels_pred[i][j]]
                conf_matrix[class_id_true, class_id_pred] += 1
        names_rows = list(s + '_true' for s in data.labelclass_to_id.keys())
        names_columns = list(s + '_pred' for s in data.labelclass_to_id.keys())
        conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

        # compute final evaluation measures
        precision_per_class = np.zeros((data.n_tags,))
        recall_per_class = np.zeros((data.n_tags,))
        for i in range(data.n_tags):
            if conf_matrix.values[i, i] > 0:
                precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
                recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
        f1 = 2 * (precision * recall) / (precision + recall)
        return Result(precision, recall, f1)
