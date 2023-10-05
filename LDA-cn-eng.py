#data prepare
article=[]
with open('final.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        article.append(line.strip())
#drop ''
for line in article:
    if line =='':
        article.remove(line)
#目前不關心的資訊:標題等相關的敘述、如作者等等...
article=article[17:]
#移除cite的相關資訊
article=article[:904]


import spacy

#英文的處理工具
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()

#將資料tokenize
def tokenize(text):
    #準備一list
    lda_tokens=[]
    #透過spacy的English模型 對text處理
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens 

import nltk
from nltk.corpus import wordnet as wn
def get_lemma(word):
    """
    >>> print(wn.morphy('dogs'))
    dog
    >>> print(wn.morphy('churches'))
    church
    >>> print(wn.morphy('aardwolves'))
    aardwolf
    """
    lemma = wn.morphy(word)
    
    if lemma is None:
        #返回原本的字
        return word
    else:
        
        return lemma
    
#stopword的移除
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    #將文字tokenize
    tokens = tokenize(text)
    #只要長度>4
    tokens =[token for token in tokens if len(token)>4]
    #這些字不能在stopword
    tokens =[token for token in tokens if token not in en_stop]
    return tokens

text_data=[]
for line in article:
    tokens =prepare_text_for_lda(line)
    #lemmatization
    tokens=[get_lemma(w) for w in tokens]
    text_data.append(tokens)
text_data = [w for w in text_data if w !=[]]

from gensim import corpora
data = text_data
#透過gensim以text_data建立字典
dictionary =corpora.Dictionary(data)
#語料庫
corpus = [dictionary.doc2bow(text) for text in data]

import pickle
pickle.dump(corpus,open('corpus.pkl','wb'))
dictionary.save('dictionary.gensim')
#透過LDA找到5個 topics
i=1
import gensim
NUM_TOPICS  =5
ldamodel = gensim.models.ldamodel.LdaModel(\
    corpus,num_topics = NUM_TOPICS ,id2word=dictionary,passes=15)
ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print("第"+str(i)+"個主題:")
    print(topic)
    i=i+1
    print()
    
    
new_doc ='treatment'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
#導入有5個主題的LDA topic model
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda,corpus,dictionary,sort_topics=False)
pyLDAvis.display(lda_display)