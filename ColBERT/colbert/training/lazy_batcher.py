import os
import ujson

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

# from colbert.utils.runs import Run


class LazyBatcher(): # patch here
    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []
        
        pids_list = []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[:self.nway]
            

            query = int(query) # here patch
            
            query = self.queries[query]
            
            
            # try:
            #     pids, scores = zipstar(pids)
            # except Exception as e:
            #     scores = []
            # patch here
            
            scores = []
                

            passages = [self.collection[int(pid)] for pid in pids] # here patch，pids有俩元素
            pids_list.append(pids)
            
            
            

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)
            
            
            all_scores = [] # patch here
            
       
        assert len(all_scores) in [0, len(all_passages)], f'{len(all_scores)}, {len(all_passages)}'
        
        
        all_scores = [int(item) for item in all_scores]  # all patch
        
        batch_data = self.collate(all_queries, all_passages, all_scores)
        
        result = []
        for i in batch_data[0]:
            result.append(i)
        result.append(pids_list)
        result = tuple(result)
        result = [result]
        # print(batch_data)
        # for i in batch_data[0]:
        #     print(type(i[0]))
        #     print(type(i[1]))
        #     print(i[0].shape)
        #     print(i[1].shape)

        # assert 1==2

        return  result# patch here

    def collate(self, queries, passages, scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx


