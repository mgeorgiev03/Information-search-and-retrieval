import collections

class BPE:

    symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            '_', "А", "а", "Б", "б", "В", "в", "Г", "г", "Д", "д", "Е", "е", 
            "Ж", "ж", "З", "з", "И", "и", "Й", "й", "К", "к", "Л", "л", "М", "м", "Н", 
            "н", "О", "о", "П", "п", "Р", "р", "С", "с", "Т", "т", "У", "у", "Ф", "ф", 
            "Х", "х", "Ц", "ц", "Ч", "ч", "Ш", "ш", "Щ", "щ", "Ъ", "ъ", "Ь", "ь", "Ю", 
            "ю", "Я", "я", ",", ".", "</w>"]
    
    def getVocab(self, corpus, num_merges): 
        raw_token_freqs = self.getDictFreqs(corpus)
        token_freqs = {}
        
        for token, _ in raw_token_freqs.items():
            token_freqs[' '.join(list(token))] = raw_token_freqs[token]

        
        for _ in range(num_merges):
            max_freq_pair = self.get_max_freq_pair(token_freqs)
            token_freqs = self.merge_symbols(max_freq_pair, token_freqs, self.symbols)
    
    def getDictFreqs(self, corpus):
        freqs = {}

        for s in corpus:
            for w in s:
                if w in freqs: freqs[w] += 1
                else: freqs[w]=1        

        temp = { w:freqs[w] for w in freqs if freqs[w] >= 1}
        
        bpe_vocab = {}
        for word, freq in temp.items():
            formatted_word = tuple(list(word) + ["</w>"])
            bpe_vocab[formatted_word] = freq

        return bpe_vocab 

    def get_max_freq_pair(self, token_freqs):
    
        pairs = collections.defaultdict(int)

        for token, freq in token_freqs.items():
            symbols = token.split()

            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        return max(pairs, key=pairs.get)  

    def merge_symbols(self, max_freq_pair, token_freqs, symbols):

        symbols.append(''.join(max_freq_pair))
        new_token_freqs = dict()

        for token, _ in token_freqs.items():

            new_token = token.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
            new_token_freqs[new_token] = token_freqs[token]

        return new_token_freqs
    
    # def segment_BPE(self, tokens):
    #     outputs = []
    #     for token in tokens:
    #         start, end = 0, len(token)
    #         cur_output = []
    #         # Segment token with the longest possible subwords from symbols
    #         while start < len(token) and start < end:
    #             if token[start: end] in self.symbols:
    #                 cur_output.append(token[start: end])
    #                 start = end
    #                 end = len(token)
    #             else:
    #                 end -= 1
    #         if start < len(token):
    #             cur_output.append('[UNK]')
    #         outputs.append(' '.join(cur_output))
    #    return outputs

    def segment_BPE(self, tokens):

        outputs = []
        for token in tokens: # tokens: '<S>', 'Според', 'мен', 'то', 'не', 'беше', 'много', 'ясно.', '<TRANS>'
            start, end = 0, len(token)
            # cur_output = []
            
            while start < len(token) and start < end:

                if token[start: end] in self.symbols:
                    
                    #cur_output.append(token[start: end])
                    outputs.append(token[start: end])
                    start = end
                    end = len(token)
                else:
                    end -= 1
                
        return outputs
        
    def tokenizeBG(self, sent):
        
        if type(sent) == "<class 'str'>":
            sent = sent.split()
        
        s = []
        for w in sent:
            if ',' in w:
                n = w.replace(',', '')
                s.append(n)
                s.append(',')
            elif '.' in w:
                n = w.replace('.', '')
                s.append(n)
                s.append('.')
            else:
                s.append(w)

        sent = s


        s = [ w + "</w>" for w in sent]
        s = ["<S>"] + self.segment_BPE(s[1:])
        return s

    def tokenizeBGEN(self, sent):

        if type(sent) == "<class 'str'>":
            sent = sent.split()

        # s = []
        # for w in sent:
        #     if ',' in w:
        #         n = w.replace(',', '')
        #         s.append(n)
        #         s.append(',')
        #     elif '.' in w:
        #         n = w.replace('.', '')
        #         s.append(n)
        #         s.append('.')
        #     else:
        #         s.append(w)

        # sent = s

        transIdx = sent.index("<TRANS>")
        s = sent[:transIdx]
        t = sent[transIdx + 1:]

        s = [ w + "</w>" for w in s]
        t = [ w + "</w>" for w in t] 
        
        s = self.segment_BPE(s)
        t = self.segment_BPE(t)

        s = s + ["<TRANS>"] + t
        return s
    