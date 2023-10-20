!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json

api_token = {"username":"bianggulugulu","key":"76e83c91bf95ebdec4ec26c78f664118"}

import json

with open('/root/.kaggle/kaggle.json', 'w+') as file:
    json.dump(api_token, file)

!chmod 600 ~/.kaggle/kaggle.json

!pip install -q kaggle

!kaggle competitions download -c 9727-assn2-22t2

!unzip 9727-assn2-22t2.zip
!pip install Sastrawi
!pip install tensorflow-text

import tensorflow as tf
from torch import dropout
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import pandas as pd


data = pd.read_csv('recipes_info.csv',dropout =true)
print(data.shape)
data.head(2)
# bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# bert is sequential based model

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/3")
!nvidia-smi

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)


model = tf.keras.Model(inputs=[text_input], outputs = [outputs])

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile()

y = model.predict(data['tags'], batch_size=32)[0]['default']

from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA


pca = PCA(n_components=100)
pca.fit(y)
y_pca = pca.transform(y)

data['vec'] = y_pca.tolist()
  

kmeans = KMeans(n_clusters=40)  
kmeans.fit(y_pca)

data['cluster'] = kmeans.fit_predict(data['vec'].values.tolist())
data['cluster']

test_data = pd.read_csv('test.csv')
clusters = data[['id','cluster']]
cluster_2_recipe_id = clusters.groupby('cluster')['id'].apply(list)
recipe_id_test = test_data['recipe_id'].to_list()

recommend_recipe = []

for id in recipe_id_test:
  find_flag = False
  for c in cluster_2_recipe_id:
    if id in c:
      try:
        recommend_recipe.append(c[0] if id!=c[0] else c[1])
      except:
        recommend_recipe.append(c[0])
      find_flag = True
      break
  if not find_flag:
    recommend_recipe.append(0)
print(len(recommend_recipe))

submission_data = pd.DataFrame([i+1 for i in range(len(recommend_recipe))], columns=['Id'])
submission_data['Predicted'] = recommend_recipe

submission_data.to_csv('submission.csv', index=False)
