class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' ', num_query=10, all_concat=False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

        self.num_query = num_query
        self.all_concat = all_concat

    def __call__(self, example):
        qry_text = example['query']

        query = self.tokenizer.encode(qry_text,
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']

            if self.all_concat:
                text = " ".join(pos['queries'][:self.num_query]) + self.tokenizer.sep_token + text
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']

            if self.all_concat:
                text = " ".join(neg['queries'][:self.num_query]) + self.tokenizer.sep_token + text
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
        qry_text = example['query']
        query = self.tokenizer.encode(qry_text,
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' ', num_query=10, all_concat=False):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

        self.num_query = num_query
        self.all_concat = all_concat

    def __call__(self, example):
        docid = example['docid']
        raw_text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(raw_text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        if self.all_concat:
            # print("Concatenating all queries...") 
            raw_text = " ".join(example['queries'][:self.num_query]) + self.tokenizer.sep_token + raw_text
            text = self.tokenizer.encode(
                raw_text,
                add_special_tokens=False,
                max_length=self.text_max_length,
                truncation=True
            )
        return {'text_id': docid, 'text': text}
