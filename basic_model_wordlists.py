# the packages numpy, tqdm, pandas have to be installed to be able to run this script.
# how to install packages with anaconda: https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/
import json
import numpy as np
import pandas as pd
import random


def load_data(filename) -> object:
    with open(filename, 'r', encoding='utf8') as infile:
        data = json.load(infile)
        data_new = dict()
        for k, v in list(data.items()):
            v_new = data_new.setdefault(k, dict())
            for k1, v1 in list(v.items()):
                v1_new = v_new.setdefault(k1, dict())
                if k1 != 'tokens':
                    for k2, v2 in list(v1.items()):
                        v2_new = [l.replace('B_', '').replace('I_', '') for l in v2]
                        v1_new[k2] = v2_new
                else:
                    v_new[k1] = v1
    return data_new


# import data
# specify the type of information which shall be extracted
# extraction_of = 'contexts'
extraction_of = 'sentiments'
# extraction_of = 'aspects'


# specify filenames in the next line
if extraction_of in ['sentiments', 'aspects']:
    filename = r'./data_laptop_absa.json'
    # filename = r'../Labeling/WiSe2020-21/export inception/data_movie_absa.json'

example_data = load_data(filename)

# possible preprocessing: lowercasing of tokens
for i, (k, v) in enumerate(example_data.items()):
    tokens = v.get('tokens')
    tokens = [token.lower() for token in tokens]
    example_data[k]['tokens'] = tokens

# train (validation) test split
# could also be extended with crossvalidation evaluation

keys = list(example_data.keys())
random.seed(1234567890)
random.shuffle(keys)

split_parameter = round(len(keys) * 0.7)
keys_train = keys[:split_parameter]
keys_test = keys[split_parameter:]

train_tokens = [example_data[k]['tokens'] for k in keys_train]
train_labels = list()
train_labels_uncertainty = list()
for k in keys_train:
    curr_users = [s for s in example_data[k].keys() if s != 'tokens']
    # for illlustration only the annotation of one user is used here -> curr_users[0]
    train_labels.append(example_data[k][curr_users[0]][extraction_of])
    train_labels_uncertainty.append(example_data[k][curr_users[0]][extraction_of + '_uncertainty'])
test_tokens = [example_data[k]['tokens'] for k in keys_test]
test_labels = list()
test_labels_uncertainty = list()
for k in keys_test:
    curr_users = [s for s in example_data[k].keys() if s != 'tokens']
    # for illlustration only the annotation of one user is used here -> curr_users[0]
    test_labels.append(example_data[k][curr_users[0]][extraction_of])
    test_labels_uncertainty.append(example_data[k][curr_users[0]][extraction_of + '_uncertainty'])

all_labelclasses = set()
for ds in [train_labels, test_labels]:
    for row in ds:
        all_labelclasses.update(row)
all_labelclasses = list(all_labelclasses)
all_labelclasses.sort()

labelclass_to_id = dict(zip(all_labelclasses, list(range(len(all_labelclasses)))))

n_tags = len(list(labelclass_to_id.keys()))

# start classification

# train classifier -> generate word lists per labelclass
# count frequencies of labelclasses per term
term_to_labelclass_to_freq = dict()
for idx, tokens in enumerate(train_tokens):
    labels = train_labels[idx]
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
# predict labels of test data (important: use only test_tokens, and do not use test_labels at all!!)
test_labels_pred = list()
for tokens in test_tokens:
    labels_pred = list()
    for tok in tokens:
        # classify token with mapping by term_to_most_freq_labelclass
        if tok in term_to_most_freq_labelclass.keys():
            labels_pred.append(term_to_most_freq_labelclass[tok])
        else:
            labels_pred.append('O')
    test_labels_pred.append(labels_pred)
# compute confusion matrix
conf_matrix = np.zeros((n_tags, n_tags))
for i, tokens in enumerate(test_tokens):
    for j, _ in enumerate(tokens):
        class_id_true = labelclass_to_id[test_labels[i][j]]
        class_id_pred = labelclass_to_id[test_labels_pred[i][j]]
        conf_matrix[class_id_true, class_id_pred] += 1
names_rows = list(s + '_true' for s in labelclass_to_id.keys())
names_columns = list(s + '_pred' for s in labelclass_to_id.keys())
conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

# compute final evaluation measures
precision_per_class = np.zeros((n_tags,))
recall_per_class = np.zeros((n_tags,))
for i in range(n_tags):
    if conf_matrix.values[i, i] > 0:
        precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
        recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])
precision = np.mean(precision_per_class)
recall = np.mean(recall_per_class)
f1 = 2 * (precision * recall) / (precision + recall)

print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('F1-measure: ' + str(f1))
