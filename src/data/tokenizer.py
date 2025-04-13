import json
import re
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0
        self.max_len = 0
        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    
    def build_vocab(self, formulas, min_freq=2):
        # Count tokens
        token_counter = Counter()
        for formula in formulas:
            tokens = self._tokenize_formula(formula)
            token_counter.update(tokens)
        
        # Build vocabulary
        self.token2id = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add frequent tokens
        for token, count in token_counter.items():
            if count >= min_freq and token not in self.token2id:
                self.token2id[token] = len(self.token2id)
        
        # Create inverse mapping
        self.id2token = {i: token for token, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)
    
    def _tokenize_formula(self, formula):
        # Simple tokenizer - improve this for better performance
        tokens = []
        for token in re.findall(r"\\[a-zA-Z]+|\S", formula):
            tokens.append(token)
        return tokens
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = self._tokenize_formula(text)
            seq = [self.token2id.get(token, self.token2id['<unk>']) for token in tokens]
            # Add start/end tokens
            seq = [self.token2id['<start>']] + seq + [self.token2id['<end>']]
            sequences.append(seq)
            self.max_len = max(self.max_len, len(seq))
        return sequences
    
    def pad_sequences(self, sequences):
        padded = np.zeros((len(sequences), self.max_len))
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'token2id': self.token2id,
                'id2token': self.id2token,
                'vocab_size': self.vocab_size,
                'max_len': self.max_len
            }, f)
    
    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.token2id = data['token2id']
        self.id2token = {int(k): v for k, v in data['id2token'].items()}
        self.vocab_size = data['vocab_size']
        self.max_len = data['max_len']
    
    def tokens_to_latex(self, tokens):
        return ' '.join(tokens).replace(' <end>', '')