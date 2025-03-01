

class TrieNode:
    def __init__(self):
        self.children = {}  
        self.is_end_of_word = False  

class Trie:
    def __init__(self):
        self.root = TrieNode()  

    def insert(self, word):
        node = self.root  
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  
            node = node.children[char]  
        node.is_end_of_word = True 

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None  
            node = node.children[char]  
        return node 
    def get_allowed_tokens(self, prefix, tokenizer = None):
        node = self.search(prefix)
        if node is None:
            return []

        allowed_tokens = []

        def dfs(current_node, current_prefix):
            if current_node.is_end_of_word:
                allowed_tokens.append(current_prefix)
            for char, next_node in current_node.children.items():
                dfs(next_node, current_prefix + char)

        dfs(node, prefix)
        # return allowed_tokens
        
        return [tokenizer.encode(word, add_special_tokens=False)[0] for word in allowed_tokens]

