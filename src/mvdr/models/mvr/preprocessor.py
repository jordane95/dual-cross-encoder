class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' ', num_view=10):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

        self.prefix_view_tokens = " ".join([self.tokenizer.decode(i) for i in range(2, 2+num_view)])

    def __call__(self, example):
        qry_text = self.tokenizer.decode(1) + self.separator + example['query']

        query = self.tokenizer.encode(qry_text,
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            text = self.prefix_view_tokens + self.separator + text
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            text = self.prefix_view_tokens + self.separator + text
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
        qry_text = self.tokenizer.decode(1) + " " + example['query']
        query = self.tokenizer.encode(qry_text,
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' ', num_query=10):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

        self.prefix_view_tokens = " ".join([self.tokenizer.decode(i) for i in range(2, 2+num_query)])

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.prefix_view_tokens + self.separator + text
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}