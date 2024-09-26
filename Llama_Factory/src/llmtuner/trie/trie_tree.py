

class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点的字典，键为字符，值为TrieNode
        self.is_end_of_word = False  # 表示是否为单词的结尾

class Trie:
    def __init__(self):
        self.root = TrieNode()  # Trie树的根节点

    def insert(self, word):
        node = self.root  # 从根节点开始插入
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  # 如果子节点不存在，创建新节点
            node = node.children[char]  # 移动到子节点，移动指针
        node.is_end_of_word = True  # 单词插入完成，标记结尾节点

    def search(self, word):
        node = self.root  # 从根节点开始查找
        for char in word:
            if char not in node.children:
                return None  # 如果某个字符不存在子节点，则前缀不匹配
            node = node.children[char]  # 移动到子节点
        return node  # 返回匹配到的最后一个节点

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
        
        # 将允许的单词转换为token ids
        return [tokenizer.encode(word, add_special_tokens=False)[0] for word in allowed_tokens]



# if __name__ == "__main__":
#     test_words = [ 'hello','dsa','hgsjiou','gyvsa','oqigugqd','8rewh']
#     trie_tree = Trie()
#     for word in test_words:
#         trie_tree.insert(word)
#     allowed_tokens = trie_tree.get_allowed_tokens('h')
#     print(allowed_tokens)
    