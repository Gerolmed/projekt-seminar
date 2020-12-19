class Data:
    """The data split into test and training"""
    def __init__(self, train_tokens, train_labels, test_tokens, test_labels, n_tags, labelclass_to_id):
        self.train_tokens = train_tokens
        self.train_labels = train_labels
        self.test_tokens = test_tokens
        self.test_labels = test_labels
        self.labelclass_to_id = labelclass_to_id
        self.n_tags = n_tags
