import torch

from tqdm import tqdm

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager

from colbert.modeling.colbert import ColBERT
import ast
import numpy as np
import json



def adjust_shape(tensor, target_shape):
    current_shape = tensor.size()
    
    # Check if the second dimension matches the target shape
    if current_shape[1] != target_shape[1]:
        # Calculate the amount of zero padding needed
        pad_amount = target_shape[1] - current_shape[1]
        
        # Pad the tensor with zeros
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_amount, 0, 0))
    
    return tensor


def resize_array(arr, new_shape=(220, 220)):
    
    original_shape = arr.shape

    # Initialize the new array with zeros
    new_array = np.zeros(new_shape)

    # Calculate the indices to copy from the original array
    copy_indices = (min(original_shape[0], new_shape[0]), min(original_shape[1], new_shape[1]))

    # Copy / crop the original array into the new array
    new_array[:copy_indices[0], :copy_indices[1]] = arr[:copy_indices[0], :copy_indices[1]]

    return new_array

class Checkpoint(ColBERT):
    """
        Easy inference with ColBERT.

        TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name, colbert_config=None, verbose:int = 3):
        super().__init__(name, colbert_config)
        assert self.training is False

        self.verbose = verbose

        self.query_tokenizer = QueryTokenizer(self.colbert_config, verbose=self.verbose)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        self.amp_manager = MixedPrecisionManager(True)
        
        self.ast_data = []
        
        # self.GCN = super().GCN()

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)

                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, context=context, bsize=bsize, full_length_search=full_length_search)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=context, full_length_search=full_length_search)
        return self.query(input_ids, attention_mask)


    def read_json(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        code_list = []
        for i in data:
            code_list.append(i['instruction'])
        count = 0
        error_num = 0
        result = {}
        for code in tqdm(code_list):
            try:
                tree = ast.parse(code)
                node_count = self.count_nodes(tree)
                adjacency_matrix = self.build_adjacency_matrix(tree,node_count)
            except:
                adjacency_matrix = np.zeros((1, 1))
                error_num += 1

            result[count] = adjacency_matrix
            count += 1
        return result

    def count_nodes(self,node):
        return 1 + sum(self.count_nodes(child) for child in ast.iter_child_nodes(node))

    def build_adjacency_matrix(self,node, node_count, mapping=None, matrix=None, parent_index=-1):
        if mapping is None:
            mapping = {}
        if matrix is None:
            matrix = np.zeros((node_count, node_count), dtype=int)

        node_index = mapping.setdefault(node, len(mapping))

        if parent_index != -1:
            matrix[parent_index, node_index] = 1

        for child in ast.iter_child_nodes(node):
            self.build_adjacency_matrix(child, node_count, mapping, matrix, node_index)

        return matrix
    
    
    def docFromText(self, docs,sampled_pids, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']
        if len(self.ast_data) == 0:
            self.ast_data = self.read_json('data/python_test/python_ast.json')
        ast_list = []
        for i in sampled_pids:
            ast_list.append(resize_array(self.ast_data[i]) )
        ast_list = torch.tensor(np.array(ast_list))
        print(ast_list.shape,'ast_list')
        # print(len(ast_list))
        # print(ast_list[0].shape)
        # print((sampled_pids[0]),'sampled_pids') # 5
        # print((docs[0]),'docs') # 代码
        
        
        
        
        
       
        # print('docs:')
        # print(len(docs))
        # error_num = 0
        # for i in docs:
        #     try:
        #         tree = ast.parse(i)
        #     except:
        #         error_num += 1
        if bsize:
            
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            returned_text = []
            if return_tokens: # False
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            # print("text_batches,here")
            # print("-"*100)
            # print(len(text_batches),'len(text_batches)')
            # for i in text_batches:
            #     print(i[0].shape, i[1].shape)
            #     print(self.doc_tokenizer.doc_maxlen,'self.doc_tokenizer.doc_maxlen')
            # print("text_batches, above")
            
            
            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                       for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

            if keep_dims is True: # False
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == 'flatten': # True
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]
                gcn_output = super().gcn(D,ast_list.to(D.device).half())
                
                D = gcn_output
                # D = D + gcn_output
                
                
                if mask.shape[1] != 220:
                    print(mask.shape)
                    print("8"*100)
                    mask = adjust_shape(mask, (mask.shape[0],220, mask.shape[2]))
                
                
                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)
                
                D = D[mask.bool().flatten()].cpu()
                
                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)

        assert False, "Implement scoring"

    def score(self, Q, D, mask=None, lengths=None):
        assert False, "Call colbert_score"
        # EVENTUALLY: Just call the colbert_score function!

        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.device).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output


"""
TODO:

def tokenize_and_encode(checkpoint, passages):
    embeddings, token_ids = checkpoint.docFromText(passages, bsize=128, keep_dims=False, showprogress=True, return_tokens=True)
    tokens = [checkpoint.doc_tokenizer.tok.convert_ids_to_tokens(ids.tolist()) for ids in token_ids]
    tokens = [tokens[:tokens.index('[PAD]') if '[PAD]' in tokens else -1] for tokens in tokens]
    tokens = [[tok for tok in tokens if tok not in checkpoint.skiplist] for tokens in tokens]

    return embeddings, tokens

"""
