import time
import torch
import random
import javalang
import torch.nn as nn
from javalang.ast import Node

from anytree import AnyNode, RenderTree
from anytree.iterators import PreOrderIter
import numpy as np
import csv
import ast
import json
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


 
def get_token(node):
    if isinstance(node, str):
        return node
    elif isinstance(node, set):
        return 'Modifier'
    elif isinstance(node, Node):
        return node.__class__.__name__
    return ''

def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
    return list(expand(children))

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)

def count_nodes(node):
    return 1 + sum(count_nodes(child) for child in ast.iter_child_nodes(node))

def build_adjacency_matrix(node, node_count, mapping=None, matrix=None, parent_index=-1):
    if mapping is None:
        mapping = {}
    if matrix is None:
        matrix = np.zeros((node_count, node_count), dtype=int)

    node_index = mapping.setdefault(node, len(mapping))

    if parent_index != -1:
        matrix[parent_index, node_index] = 1

    for child in ast.iter_child_nodes(node):
        build_adjacency_matrix(child, node_count, mapping, matrix, node_index)

    return matrix
# def read_tsv(filename):
#     result = []
#     with open(filename, 'r', encoding='utf-8') as file:
#         tsv_reader = csv.reader(file, delimiter='\t')
#         for row in tsv_reader:
#             print(type(row))
#             print(len(row))
#             print(row[0])
#             print(row[1])
#             code = row[1]
#             parsed_tree = ast.parse(code)
#             print(parsed_tree)
#             quit()
#             result.append(row)
#     return result

def convert_java_ast(code_str):
    code_str = 'public class MyClass { ' + code_str + '} '
    # try:
    programtokens=javalang.tokenizer.tokenize(code_str)
    parsed_tree = javalang.parse.Parser(programtokens)
    programast=parsed_tree.parse_member_declaration()

    # except Exception as e:
        
    #     return -1
    
    tree = programast
    nodelist = []
    newtree=AnyNode(id=0,token=None,data=None)
    createtree(newtree, tree, nodelist)
    newtree = build_adjacency_matrix_java(newtree)
    
    return newtree

def build_adjacency_matrix_java(root):
    nodes = list(PreOrderIter(root))
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for node in nodes:
        for child in node.children:
            i, j = index[node], index[child]
            adjacency_matrix[i][j] = 1
    
    return adjacency_matrix

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    code_list = []
    for i in data:
        code_list.append(i['instruction'])
    count = 0
    result = {}
    for code in tqdm(code_list):
        if 'python' in filename:
            try:
                tree = ast.parse(code)
                node_count = count_nodes(tree)
                adjacency_matrix = build_adjacency_matrix(tree,node_count)
            except:
                adjacency_matrix = np.zeros((1, 1))
        elif 'java' in filename:
                # print("true")
            try:
                adjacency_matrix = convert_java_ast(code)
            except:
                print("error")
                adjacency_matrix = np.zeros((1, 1))
                # print(adjacency_matrix.shape)
            # except Exception as e:
            #     # print("false")
            #     print(e)
            #     adjacency_matrix = np.zeros((1, 1))
        else:
            assert 1==2

        result[str(count)] = adjacency_matrix
        count += 1
    
    return result


def train(config: ColBERTConfig, triples, queries=None, collection=None,python_ast_file ='data/python_train/python_ast.json'):
    
    
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])
    count_out_looper = 0
    count_in_looper = 0
        
    ast_data = read_json(python_ast_file)

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        
        count_out_looper += 1
        
        
        
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            count_in_looper += 1
            with amp.context():
                try:
                    queries, passages, target_scores, pids_list = batch
                    
                    
                    for id,value in enumerate(pids_list):
                        # print(id, value)
                        pids_list[id][0] = ast_data[value[0]]
                        pids_list[id][1] = ast_data[value[1]]

                    
                    
                    # queries, passages, target_scores = batch
                    encoding = [queries, passages, pids_list]
                    
                    
                    
                except Exception as e:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]
                    print(f'The Exception is {e}')
           
                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        # if config.rank < 1:
        #     print_message(batch_idx, train_loss)
        #     manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    # if config.rank < 1:
    #     print_message("#> Done with all triples!")
    #     ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)
        if config.rank < 1 and count_out_looper % 100 == 0:
            print_message("#> Done with all triples!")
            ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

    return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.



def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
