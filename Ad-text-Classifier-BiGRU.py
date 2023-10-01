import feather
import os
import re
import sys  
import gc
import random
import pandas as pd
import numpy as np
import jieba
from sklearn.utils import delayed, Parallel

args = {
"gpu" : "0",
"column_name" : "word_content",
"word_seq_len" : 600,
"embedding_vector" : 300,

"num_words" : 100000,
"model_name" : "bi_gru_model",
"batch_size" : 256,
"KFold" : 6,
"classification" : 2
}
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from scipy import stats
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.engine.topology import Layer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict( get_coefs( *line.strip().split(' ') ) for line in f )
    
model_23 = load_embeddings('../../中文词向量/Tencent_AILab_ChineseEmb.txt')

if not os.path.exists("../embedding"):
    os.mkdir("../embedding")

if not os.path.exists("../cache"):
    os.mkdir("../cache")

if not os.path.exists("../stacking"):
    os.mkdir("../stacking")

if not os.path.exists("../mid_result"):
    os.mkdir("../mid_result")

if not os.path.exists("../submission"):
    os.mkdir("../submission")

train = pd.read_csv('../input/competeDataForA.csv',encoding='utf-8',sep='\t')
test = pd.read_csv('../input/evaluationDataForA.csv',encoding='utf-8',sep='\t')

def cleaner(x):
    
    # 只提取所有的中文
    # regex = re.compile('[\u4e00-\u9fa5]+')
    # x = regex.findall(x)
    # if x == "nan" or x == "" or x == None:
    #     x = "空"
    # x = "".join(x)
    
    x = x.replace('NO.1', '第一').replace('NO.2', '第二').replace('NO.3', '第三')
    
    # 剔除所有的标点符号
    x = re.sub("[\s+\\\\!\/,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", x)
    
    return x.lower()

train['ocr'] = Parallel(n_jobs=20)(delayed(cleaner)(x) for x in train.ocr.values)
test['ocr']  = Parallel(n_jobs=20)(delayed(cleaner)(x) for x in test.ocr.values)

def cut(x):
    return " ".join(list(jieba.cut(x)))

train['word_content'] = Parallel(n_jobs=40)(delayed(cut)(x) for x in train.ocr.values)
test['word_content']  = Parallel(n_jobs=40)(delayed(cut)(x) for x in test.ocr.values)

#词向量
def w2v_pad(df_train,df_test,col, maxlen_,victor_size):

    tokenizer = text.Tokenizer(num_words=args["num_words"], lower=False,filters="")
    tokenizer.fit_on_texts(list(df_train[col].values)+list(df_test[col].values))

    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].values), maxlen=maxlen_)
    test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].values), maxlen=maxlen_)
    
    word_index = tokenizer.word_index
    
    count = 0
    nb_words = len(word_index)
    print(nb_words)
    all_data=pd.concat([df_train[col],df_test[col]])
           
    embedding_word2vec_matrix = np.zeros((nb_words + 1, 200))
    for word, i in word_index.items():
        embedding_vector = model_23[word] if word in model_23 else None
        if embedding_vector is not None:
            count += 1
            embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(200) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_word2vec_matrix[i] = unk_vec

    print (embedding_word2vec_matrix.shape, train_.shape, test_.shape)
    return train_, test_, word_index, embedding_word2vec_matrix

word_seq_len=args["word_seq_len"]
victor_size=args["embedding_vector"]
column_name=args["column_name"]
train_, test_,word2idx, word_embedding = w2v_pad(train,test,column_name, word_seq_len,victor_size)

def bi_gru_model(sent_length, embeddings_weight,class_num):
    print("bi_gru_model")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(400, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def word_model_cv(my_opt):
    #参数
    lb = LabelEncoder()
    train_label = lb.fit_transform(train['label'].values)
    train_label = to_categorical(train_label)

    if not os.path.exists("../cache/"+my_opt):
        os.mkdir("../cache/"+my_opt)

    my_opt=eval(my_opt)
    name = str(my_opt.__name__)
    kf = KFold(n_splits=args["KFold"], shuffle=True, random_state=520).split(train_)
    train_model_pred = np.zeros((train_.shape[0], args["classification"]))
    test_model_pred = np.zeros((test_.shape[0], args["classification"]))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = train_[train_fold, :], train_[test_fold, :]
        y_train, y_valid = train_label[train_fold], train_label[test_fold]

        print(i, 'fold')

        the_path = '../cache/' + name +'/' +  name + "_" +args["column_name"]
        model = my_opt(word_seq_len, word_embedding,args["classification"])
        model.summary()
        early_stopping = EarlyStopping(monitor='val_acc', patience=6)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(the_path + str(i) + '.hdf5', monitor='val_acc', verbose=2, save_best_only=True, mode='max',save_weights_only=True)

        model.fit(X_train, y_train,
                  epochs=30,
                  batch_size=args["batch_size"],
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping, plateau, checkpoint],
                  verbose=1)
        model.load_weights(the_path + str(i) + '.hdf5')


        print (name + ": valid's accuracy: %s" % f1_score(lb.inverse_transform(np.argmax(y_valid, 1)), 
                                                          lb.inverse_transform(np.argmax(model.predict(X_valid), 1)).reshape(-1,1),
                                                          average='micro'))
    
        train_model_pred[test_fold, :] =  model.predict(X_valid)
        test_model_pred += model.predict(test_)
        
        del model; gc.collect()
        K.clear_session()

    print (name + ": offline test score: %s" % f1_score(lb.inverse_transform(np.argmax(train_label, 1)), 
                                                  lb.inverse_transform(np.argmax(train_model_pred, 1)).reshape(-1,1),
                                                  average='micro'))

    last_pred=test[['id']].copy()
    last_pred['proba'] = test_model_pred[:,1]
    last_pred['label']=lb.inverse_transform(np.argmax(test_model_pred, 1)).reshape(-1,1)
    last_pred[['id',"label"]].to_csv('../submission/submit.csv',index=False)

word_model_cv(args["model_name"])
pd.read_csv('../submission/submit.csv')[['id',"label"]].to_csv('../submission/sub5.csv',index=False)