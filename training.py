from tensorboard import notebook


import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.activations import softmax
import nltk
from nltk.corpus import stopwords
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
urls=["https://onlymyenglish.com/simple-english-sentences-for-kids/" ,
"https://onlymyenglish.com/daily-use-english-sentences/" ,
"https://onlymyenglish.com/1000-english-sentences-used-in-daily-life/" ,
"https://onlymyenglish.com/sentences-for-students/" ,
"https://onlymyenglish.com/short-sentences/",
"https://onlymyenglish.com/different-ways-to-say-you-are-welcome/",
"https://onlymyenglish.com/ways-to-say-im-sorry/",
"https://onlymyenglish.com/ways-to-say-thank-you/",
"https://onlymyenglish.com/ways-to-say-good-night/",
"https://onlymyenglish.com/examples-of-simple-present-tense/"]
urls2=["https://sentence.yourdictionary.com/vegetables",
"https://sentence.yourdictionary.com/clothes",
"https://sentence.yourdictionary.com/food",
"https://sentence.yourdictionary.com/family",
"https://sentence.yourdictionary.com/sister",
"https://www.fluentu.com/blog/english/talking-about-food-in-english/"]

text=[]
def scrap_urls(urls,balise):
    for i in range(len(urls)):
        reponse=requests.get(urls[i])
        soup=BeautifulSoup(reponse.text,"html.parser")
        soup
        ils=soup.findAll(balise)
        for j in range (len(ils)):
            text.append(ils[j].text)
            
#scrap_urls(urls2,"p")
scrap_urls(urls,"li")

"""df = pd.read_csv('compliment.csv' , error_bad_lines=False)
csv_list = df.values.tolist()
text+= csv_list"""

joined_text=" ".join(text)
partial_text=joined_text[:]


tokenizer = RegexpTokenizer(r"\w+")
tokens=tokenizer.tokenize(partial_text.lower())

unique_tokens=np.unique(tokens)
unique_token_index={token :idx for idx, token in enumerate(unique_tokens)}

n_words=3
input_words=[]
next_words=[]
for i in range(len(tokens)-n_words):
    input_words.append(tokens[i:i+n_words])
    next_words.append(tokens[i+n_words])
    
X=np.zeros((len(input_words),n_words, len(unique_tokens)),dtype=bool)
y=np.zeros((len(next_words),len(unique_tokens)),dtype=bool)

for i, words in enumerate (input_words):
    for j , word in enumerate (words):
        X[i,j,unique_token_index[word]]=1
    y[i,unique_token_index[next_words[i]]]=1
    
model=Sequential()
model.add(LSTM(128, input_shape=(n_words,len(unique_tokens)),return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation(tf.nn.softmax))

model.compile(loss="categorical_crossentropy",optimizer=RMSprop(learning_rate=0.01),metrics=["accuracy"])

# Définir un répertoire pour enregistrer les événements
'''log_dir = "./logs"

# Créer un FileWriter pour enregistrer les événements
writer = tf.summary.FileWriter(log_dir)

# Créer un placeholder pour la perte
loss_placeholder = tf.placeholder(tf.float32, shape=())
# Ajouter un sommaire pour la perte
loss_summary = tf.summary.scalar('loss', loss_placeholder)

# Lors de l'entraînement du modèle, enregistrer les événements
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        # ... entraînement du modèle ...
        model.fit(X,y,batch_size=128,epochs=50,shuffle=True)
        # Calculer la loss
        loss = model.evaluate(X, y)[0]
        # Ajouter le sommaire pour la perte
        summary = sess.run(loss_summary, feed_dict={loss_placeholder: loss})
        writer.add_summary(summary, global_step=i)'''

# Créer les ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

file_name = 'my_saved_model'

tensorboard = TensorBoard(log_dir="log\\{}".format(file_name))

history = model.fit(X_train, y_train ,verbose=1, epochs=50, batch_size=64,
                     validation_data=(X_test, y_test), callbacks=[tensorboard])


#model.fit(X,y,batch_size=128,epochs=10,shuffle=True)

model.save("mymodel.h5")
model=load_model("mymodel.h5")

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

stop_words = set(stopwords.words('english'))

def predict_next_word(input_text,n_best):
    input_text=input_text.lower()
    X=np.zeros((1,n_words,len(unique_tokens)))
    for i , word in enumerate(input_text.split()):
        X[0,i, unique_token_index[word]]= 1
    predictions=model.predict(X)[0]
    predictions = np.argpartition(predictions, -n_best)[ -n_best:]
    # convert indices to words
    possible_words = [unique_tokens[idx] for idx in predictions]
    # remove stop words
    possible_words = [word for word in possible_words if not word.lower() in stop_words]
    return possible_words[:3]

#notebook.start("--logdir=C:/Users/ASUS/.spyder-py3/log")



            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            