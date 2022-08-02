class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            text = " ".join([example['query'], self.tokenizer.sep_token, text])
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            text = " ".join([example['query'], self.tokenizer.sep_token, text])
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' ', num_query=10):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

        self.num_query = num_query

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']

        queries = example['queries'][:self.num_query]
        
        texts = [" ".join([query, self.tokenizer.sep_token, text]) for query in queries]
        
        texts = [
            self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.text_max_length,
            truncation=True
            ) for text in texts
        ] # List[List[int]]
        return {'text_id': docid, 'text': texts}
