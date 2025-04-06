import os
os.environ["http_proxy"] = "http://127.0.0.1:33210"
os.environ["https_proxy"] = "http://127.0.0.1:33210"
from transformers import AutoTokenizer
from collections import defaultdict


def create_corpus():
    """创建语料库"""
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens."
    ]
    return corpus


def initial_tokenizer():
    """初始化分词器，来进行分词前的预处理"""
    return AutoTokenizer.from_pretrained("gpt2")


def compute_corpus_freq(corpus, tokenizer):
    """计算词频"""
    words_freq = defaultdict(int)
    for text in corpus:
        words_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        words = [word for word, offset in words_with_offset]
        for word in words:
            words_freq[word] += 1
    return words_freq


def build_initial_vocab(words_freq):
    """构建初始词汇表（基础字符+特殊标记）"""
    alphabet = []
    for word in words_freq.keys():
        for char in word:
            if char not in alphabet:    alphabet.append(char)
    vocab = ["<|endoftext|>"] + sorted(alphabet)
    return vocab


def split_words_2_chars(words_freq):
    """将单词拆分为字符"""
    return {word: list(word) for word in words_freq.keys()}


def compute_max_pair_freq(splits, words_freq, vocab):
    """计算相邻字符对的最高频率"""
    max_freq = None
    best_pair = ""
    pairs_freq = defaultdict(int)
    for word, freq in words_freq.items():
        chars = splits[word]
        if len(chars) <= 1:
            continue
        for i in range(len(chars) - 1):
            pair = chars[i] + chars[i + 1]
            if pair not in vocab:
                pairs_freq[pair] += freq
            if max_freq is None or max_freq < pairs_freq[pair]:
                max_freq = pairs_freq[pair]
                best_pair = pair
    return best_pair, max_freq


def merge_in_splits(best_pair, splits, words_freq):
    """在 splits 中合并频率最高字符对"""
    for word in words_freq:
        chars = splits[word]
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                new_chars.append(best_pair[0] + best_pair[1])
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        splits[word] = new_chars
    return splits


def train_BPE(vocab, splits, words_freq, target_vocab_size=50):
    """执行 BPE 训练直到达到目标词汇表大小"""
    merges = {}
    while len(vocab) < target_vocab_size:
        best_pair, max_freq = compute_max_pair_freq(splits, words_freq, vocab)
        if not best_pair:  # 没有更多可合并的对
            break

        splits = merge_in_splits(best_pair, splits, words_freq)
        merges[best_pair] = best_pair

        # 更新词汇表
        vocab.append(best_pair)

    return vocab, merges


def tokenize(text, tokenizer, merges):
    """使用训练得到的合并规则进行分词"""
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    words = [word for word, offset in pre_tokenize_result]
    splits = [list(word) for word in words]
    for pair, merge in merges.items():
        for i in range(len(splits)):
            chars = splits[i]
            new_chars = []
            j = 0
            while j < len(chars):
                if j < len(chars) - 1 and chars[j] == pair[0] and chars[j+1] == pair[1]:
                    new_chars.append(merge)
                    j += 2
                else:
                    new_chars.append(chars[j])
                    j += 1
            splits[i] = new_chars
    return sum(splits, [])


def main():
    # 初始化词表和 tokenizer
    corpus = create_corpus()
    tokenizer = initial_tokenizer()
    # 统计词频
    words_freq = compute_corpus_freq(corpus, tokenizer)
    print(f"word frequencies:{words_freq}")
    # 构建初始词汇表
    vocab = build_initial_vocab(words_freq)
    print(f"初始化的词表{vocab}\n初始词表的长度：{len(vocab)}")
    # 字符级拆分
    splits = split_words_2_chars(words_freq)
    # BPE 训练
    vocab, merges = train_BPE(vocab, splits, words_freq)
    print(f"vocab after training:{vocab}\nmerges: {merges}")
    # 测试分词
    text = "This section shows several tokenizer algorithms."
    tokens = tokenize(text, tokenizer, merges)
    print(f"Tokenized '{text}': {tokens}")


if __name__ == "__main__":
    main()
