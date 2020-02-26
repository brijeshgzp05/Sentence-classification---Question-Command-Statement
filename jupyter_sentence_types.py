
# coding: utf-8

# # Sentence classification - Question, Command, Statement

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import os


# ## Reading Data and displaying it.
# I have processed it already. For you raw data is provided in data file1.csv

# In[4]:


import pandas as pd
os.chdir("/content/gdrive/My Drive")

df = pd.read_csv("processed_full_spaadia.csv")
df.head()


# ## Some more exloration

# In[5]:


df.shape


# In[6]:


print(df.isnull().sum())


# ## Changing categotical type to labels 

# In[ ]:


types=df.type.unique()
dic={}
for i,type_ in enumerate(types):
    dic[type_]=i
labels=df.type.apply(lambda x:dic[x])


# ## Keras Import 

# In[8]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# ## Splitting the data through sklearn

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.statement, df.type, test_size=0.3, random_state=42)


# In[ ]:


val_data = pd.concat([X_test,y_test],axis=1)
train_data = pd.concat([X_train,y_train],axis=1)


# In[ ]:


texts=df.statement


# ## Building Tokenizer

# In[12]:


NUM_WORDS=10**5
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
tokenizer.fit_on_texts(texts)
sequences_train = tokenizer.texts_to_sequences(train_data.statement)
sequences_valid=tokenizer.texts_to_sequences(val_data.statement)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# ## Padding

# In[ ]:


import numpy as np


# In[ ]:


max_length=50
trunc_type='post'


# In[15]:


X_train = pad_sequences(sequences_train,maxlen=max_length, truncating = trunc_type)
X_val = pad_sequences(sequences_valid,maxlen=max_length, truncating = trunc_type)
y_train = to_categorical(np.asarray(labels[train_data.index]))
y_val = to_categorical(np.asarray(labels[val_data.index]))
print('Shape of X train and X validation tensor:', X_train.shape,X_val.shape)
print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)


# ## Constructing the model

# In[ ]:


from keras.optimizers import Adam
import keras


# In[17]:


EMBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
model = keras.Sequential([
    keras.layers.Embedding(vocabulary_size,EMBEDDING_DIM,input_length=max_length),
    keras.layers.LSTM(16, activation='relu',return_sequences=True),
    keras.layers.Dropout(0.25),
    keras.layers.LSTM(8, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(3, activation='softmax')
])


# ## Compiling the model

# In[18]:


adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])


# ## Summary

# In[19]:


model.summary()


# ## Creating a callback function

# In[ ]:



class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if (logs.get('acc')>logs.get('val_acc')):
      print("\nOverfitting begins")
      self.model.stop_training=True

callbacks = myCallback()


# ## Fittng the data to model

# In[21]:


epochs=10
verbose=1
batch_size=32
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
history=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_val, y_val), callbacks=[callbacks])


# In[ ]:


def prediction_single(sent):
    l = []
    l.append(sent)
    sequences_pred = tokenizer.texts_to_sequences(l)
    padded_pred = pad_sequences(sequences_pred,maxlen=max_length, truncating = trunc_type)
    pred = model.predict([padded_pred])
    pred = list(pred[0])
    ind = np.argmax(pred)
    if ind==0:
        return 'command'
    elif ind==1:
        return 'statement'
    else:
        return 'question'


# In[ ]:


typ = prediction_single("Your sentence")
print(typ)


# In[ ]:


model.save_weights("processed_full_spaadia_keras_model_new.h5")


# In[ ]:


import pickle

# saving
with open('processed_full_spaadia_keras_tokenizer_new.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# #### As you can see there is an imbalance in data i.e, of command type. But it works good on statements and questions. For commands we need more data.
