import json
import sys
import nltk
import regex
import copy


def or_operation(destination, src, op):
    res = copy.deepcopy(destination)
    for token in src:
        if token in destination:
            for doc in src[token]:
                if doc in destination[token]:
                    tobe_res = []
                    dest_i, src_i = 0, 0
                    while True:
                        if dest_i == len(destination[token][doc]):
                            if src[token][doc][src_i] not in tobe_res:
                                tobe_res.append(src[token][doc][src_i])
                            src_i += 1
                        elif src_i == len(src[token][doc]):
                            if destination[token][doc][dest_i] not in tobe_res:
                                tobe_res.append(destination[token][doc][dest_i])
                            dest_i += 1
                        elif destination[token][doc][dest_i] > src[token][doc][src_i]:
                            if src[token][doc][src_i] not in tobe_res:
                                tobe_res.append(src[token][doc][src_i])
                            src_i += 1
                        else:
                            if destination[token][doc][dest_i] not in tobe_res:
                                tobe_res.append(destination[token][doc][dest_i])
                            dest_i += 1
                        if dest_i == len(destination[token][doc]) and src_i == len(src[token][doc]):
                            break
                    res[token][doc] = tobe_res
                else:
                    res[token][doc] = src[token][doc]
        else:
            res[token] = src[token]
    return res


def all_docs(src_index):
    return list({doc for token in src_index for doc in src_index[token]})


def intersection_op(destination, src):
    """
    intersection of two documents
    :param destination:
    :param src:
    :return:
    """
    result_list = []
    src_i, destination_i = 0, 0
    while src_i < len(destination) or destination_i < len(src):
        
        if destination[src_i] > src[destination_i]:
            destination_i += 1
        elif destination[src_i] < src[destination_i]:
            src_i += 1
        else:
            result_list.append(destination[src_i])
            src_i += 1
            destination_i += 1
    return result_list


def intersection_doc(lst1, lst2):
    """
    intersection of two documents
    :param lst1:
    :param lst2:
    :return:
    """
    return list(set(lst1) & set(lst2))


def filter_docs(doc_list, selected_docs):
    """
    filter the documents according to selected docs
    :param doc_list:
    :param selected_docs:
    :return:
    """
    res = {}
    for doc in doc_list:
        if doc in selected_docs:
            res[doc] = doc_list[doc]
    return res


def and_query(destination, src, op):
    filtered_docs = intersection_doc(all_docs(destination), all_docs(src))
    res = {}
    for token in src:
        if token not in destination:
            temp_doc = filter_docs(src[token], filtered_docs)
            if temp_doc != {}:
                res[token] = temp_doc
        else:
            res[token] = {}
            for doc in src[token]:
                if doc in destination[token]:
                    left = destination[token][doc]
                    right = src[token][doc]
                    if regex.match('&', op) is not None:
                        common_position = intersection_op(left, right)
                        if common_position:
                            res[token][doc] = common_position
            if res[token] == {}:
                del res[token]
    return res


def verify(left, right, op, doc):
    if regex.match(r'\+[0-9]+', op):
        distance = int(op[1:])
        if right > left and right - left <= distance:
            return True
    if regex.match(r'/[0-9]+', op):
        distance = int(op[1:])
        if abs(right - left) <= distance:
            return True
    
    left = 0
    if doc in index['.']:
        while left < len(index['.'][doc]):
            if index['.'][doc][left] > left:
                break
            left += 1
    if left == 0:
        return True
    right = 0
    if doc in index['.']:
        while right < len(index['.'][doc]):
            if index['.'][doc][right] > right:
                break
            right += 1
    
    if left == right:
        if regex.match('/s', op):
            return True
        if regex.match(r'\+s', op):
            if right > left:
                return True
    return False


def add_index(src_index, token, doc, position):
    """
    add index to source
    :param src_index:
    :param token:
    :param doc:
    :param position:
    :return:
    """
    if token not in src_index:
        src_index[token] = {}
    if doc not in src_index[token]:
        src_index[token][doc] = set()
    src_index[token][doc].add(position)


def common_doc(doc1, doc2):
    """
    return common element of each docs
    :param doc1:
    :param doc2:
    :return:
    """
    processed_results = set()
    for token1 in doc1:
        for token2 in doc2:
            for doc in doc1[token1]:
                if doc in doc2[token2]:
                    processed_results.add(doc)
    return list(processed_results)


def evaluate(before, after, op):
    result = {}
    res_common_docs = common_doc(before, after)
    for l_token in before:
        for doc in res_common_docs:
            if doc not in before[l_token]:
                continue
            for r_token in after:
                if doc not in after[r_token]:
                    continue
                for l_pos in before[l_token][doc]:
                    for r_pos in after[r_token][doc]:
                        if verify(l_pos, r_pos, op, doc):
                            add_index(result, l_token, doc, l_pos)
                            add_index(result, r_token, doc, r_pos)
    
    return result


class Processor:
    def __init__(self):
        self.cache = {}

        terms = {
            'expression': ['&'],
            'in_sentence': ['/s'],
            'after_sentence': [r'\+s'],
            'in_n': [r'/[0-9]+'],
            'after_n': [r'\+[0-9]+'],
            'term': [r'\|'],
            'factor': [],
            'word': [],
        }

        operations = {
            'expression': and_query,
            'in_sentence': evaluate,
            'after_sentence': evaluate,
            'in_n': evaluate,
            'after_n': evaluate,
            'term': or_operation,
        }
        
        self.terms = terms
        self.term_names = list(self.terms.keys())
        self.operations = operations
    
    def value(self, text):
        porter = nltk.PorterStemmer()
        try:
            text = list(map(str.lower, [text]))
            text = list(map(porter.stem, text))
            return {text[0]: index[text[0]]}
        except:
            return {}
    
    def parse(self, token):
        self.tokens = token
        self.pos = -1
        self.len = len(token) - 1
        rv = self.first()
        self.verify_end()
        return rv
    
    def verify_end(self):
        if self.pos < self.len:
            raise RuntimeError('length error')
    
    def keyword(self, *keywords):
        if self.pos >= self.len:
            raise Exception(
                self.pos + 1,
                'Expected %s but got end of string',
                ','.join(keywords)
            )
        
        for keyword in keywords:
            if regex.match(keyword, self.tokens[self.pos + 1]) is not None:
                return self.next(False)
        
        raise Exception(self.pos + 1, 'Expected %s but got %s', ','.join(keywords), self.tokens[self.pos + 1], )
    
    def like_keyword(self, *keywords):
        try:
            return self.keyword(*keywords)
        except:
            return None
    
    def next_term(self, term):
        return self.term_names[self.term_names.index(term) + 1]
    
    def next(self, to_value=True):
        self.pos += 1
        if to_value:
            return self.value(self.tokens[self.pos])
        return self.tokens[self.pos]
    
    def dive_into(self, term):
        if len(self.terms[term]) > 0 and callable(self.terms[term][0]):
            return self.terms[term][0]()
        try:
            return getattr(self, term)()
        except:
            return self.default(term)
    
    def first(self):
        self.default = self.evaluate
        return self.default(self.term_names[0])
    
    def factor(self):
        if self.like_keyword(r'\('):
            rv = self.matching(self.term_names[0])
            self.keyword(r'\)')
            return rv
        return self.matching(self.term_names[-1])
    
    def word(self):
        return self.next()
    
    def matching(self, *rules):
        for rule in rules:
            rv = self.dive_into(rule)
            return rv
    
    def evaluate(self, term):
        next_one = self.next_term(term)
        op = self.terms[term]
        rv = self.matching(next_one)
        while True:
            op = self.like_keyword(*op)
            if op is None:
                break
            term = self.matching(next_one)
            rv = self.operate(rv, term, op)
        return rv
    
    def operate(self, dest, src, op):
        for i in self.term_names:
            for ops in self.terms[i]:
                if regex.match(ops, op):
                    return self.operations[i](dest, src, op)
    
    def handle_query(self, key_word):
        subset = [x.group() for x in regex.finditer(r'"(\w+ )*(\w+ ?)?"', key_word)]
        post = [regex.sub(r'(?<=\w+) +(?=\w+)', ' +1 ', x) for x in subset]
        for i in range(len(subset)):
            key_word = regex.sub(subset[i], "(" + post[i][1:-1] + ")", key_word, count=1)
        key_word = regex.sub(r'([()])', r" \1 ", key_word)
        key_word = regex.sub(r'(?<=(\n| |^)[^+/&( ]\w*) +(?=(\( *)* *\w+)', ' | ', key_word)
        key_word = key_word.split()
        
        answer = self.parse(key_word)
        result = set()
        for a in answer:
            for doc in answer[a]:
                result.add(int(doc))
        result = sorted(result)
        return result


if __name__ == '__main__':
    path_index = sys.argv[1]
    with open(path_index) as f:
        index = f.read()
        index = json.loads(index)
    
    parser = Processor()
    
    for line in sys.stdin:
        query = line.strip()
        results = parser.handle_query(query)
        for res in results:
            print(res)
