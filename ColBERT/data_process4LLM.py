import json
from tqdm import tqdm
import ast
import random


def split_into_subtrees(node, code_str, subtrees=None):
    if subtrees is None:
        subtrees = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) or isinstance(child, ast.For) or isinstance(child, ast.Try):
            start_pos = child.lineno - 1
            end_pos = child.end_lineno
            subtree_code = code_str.split('\n')[start_pos:end_pos]
            subtrees.append("\n".join(subtree_code))

        else:
            subtrees = split_into_subtrees(child, code_str, subtrees)

    return subtrees

def get_subtree_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    code_list = []
    
    
    code_nl_list = []
    for i in data:
        code_nl_list.append([i['instruction'],i['output']])
        code_list.append(i['instruction'])
    result = []
    count = 0

    
    for code in tqdm(code_nl_list):
        code, nl = code
        try:
            tree = ast.parse(code)
        except:
            continue
        subtrees = split_into_subtrees(tree,code)
            
        if len(subtrees) > 1:
            result.append({
                    'instruction':code,
                    'output':nl,
                    'subtrees':subtrees
                })
        else:
            result.append({
                    'instruction':code,
                    'output':nl,
                    'subtrees':[code]
                })
    random.shuffle(result)
       
            
        

    return result


def process_jsonl(subtree_data, b_filename,prompt):
    
    output_data = []
    for i in subtree_data:
        code = i['instruction']
        nl = i['output']
        subtrees = i['subtrees']
        for tree in subtrees:
            if len(tree) > 30:
                output_data.append({
                    'instruction':prompt,
                    'output':nl,
                    'tree':tree,
                    'input':code
                })
    with open(b_filename, 'w') as file:
        json.dump(output_data, file, indent=4)
        
        
def get_sft_data(json_name, write_json, prompt):
    with open(json_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    
    data = random.sample(data, 500)
    result = []
    for i in data:
        result.append({
            'input':i['instruction'],
            "history": [],
            'output':i['output'],
            'instruction':prompt
        })
    
    with open(write_json, 'w') as file:
        json.dump(result, file, indent=4)
    
    
            
if __name__ == '__main__':
    prompt = ''
    ast_data = get_subtree_data('data/python_train.json')
    process_jsonl(ast_data,'data/python_generation.json',prompt)
    get_sft_data('data/python_train.json','data/python_sft.json',prompt)
