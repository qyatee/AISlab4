import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import os
from keras.layers import Dense
from keras.layers import LSTM,Input
from keras.models import Model


batch_size = 64  # Batch size for training.
epochs = 100 # Number of epochs to train for.
latent_dim = 256 
english_texts=[]
french_texts=[]
english_character=[]
french_character=[]
with open("fra.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[:30000]:
    english_text,french_text,_=line.split("\t")
    english_texts.append(english_text)
    french_text = "\t" + french_text + "\n"
    french_texts.append(french_text)
    
for i in english_texts:
    for c in i:
        if c not in english_character:
            english_character.append(c)
            english_character.sort()

for j in french_texts:
    for c in j:
        if c not in french_character:
            french_character.append(c)
            french_character.sort()

english_d={}
for i in range(len(english_character)):
    english_d[english_character[i]]=i

french_d={}
for i in range(len(french_character)):
    french_d[french_character[i]]=i

english_encoder_tokens = len(english_character)
french_decoder_tokens = len(french_character)

max_encoder_seq_length=0
for i in english_texts:
    if len(i)>max_encoder_seq_length:
        max_encoder_seq_length=len(i)
        
max_decoder_seq_length=0
for i in french_texts:
    if len(i)>max_decoder_seq_length:
        max_decoder_seq_length=len(i)


encoder_input_data=[]
for bb in range(30000):
    a=[]
    b=[]
    c=[]
    k=len(english_texts[bb])
    m=0
    while m<k:
        for char in english_texts[bb][m]:
            for i in range(len(english_character)):
                if english_d[char]==i:
                    a.append(1)
                else:
                    a.append(0)

        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    while m<max_encoder_seq_length:
        for i in range(len(english_character)):
            if i==0:
                a.append(1)
            else:
                a.append(0)
        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    encoder_input_data.append(c)

encoder_input_data=np.array(encoder_input_data)
decoder_input_data=[]
for bb in range(30000):
    a=[]
    b=[]
    c=[]
    k=len(french_texts[bb])
    m=0
    while m<k:
        for char in french_texts[bb][m]:
            for i in range(len(french_character)):
                if french_d[char]==i:
                    a.append(1)
                else:
                    a.append(0)

        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    while m<max_decoder_seq_length:
        for i in range(len(french_character)):
            if i==0:
                a.append(1)
            else:
                a.append(0)
        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    decoder_input_data.append(c)

decoder_input_data=np.array(decoder_input_data)
decoder_target_data=[]
for bb in range(30000):
    a=[]
    b=[]
    c=[]
    k=len(french_texts[bb])
    m=1
    while m<k:
        for char in french_texts[bb][m]:
            for i in range(len(french_character)):
                if french_d[char]==i:
                    a.append(1)
                else:
                    a.append(0)

        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    m=m-1
    while m<max_decoder_seq_length:
        for i in range(len(french_character)):
            if i==0:
                a.append(1)
            else:
                a.append(0)
        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    decoder_target_data.append(c)
#--
decoder_target_data=np.array(decoder_target_data)
encoder_inputs = Input(shape=(None,len(english_character)))
encoder = LSTM(latent_dim,dropout=0.2,return_sequences=True,return_state=True)
encoder_outputs_1, state_h_1, state_c_1 = encoder(encoder_inputs)
encoder = LSTM(latent_dim,dropout=0.2,return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_outputs_1)
encoder_states = [state_h_1,state_c_1,state_h, state_c]

decoder_inputs = Input(shape=(None, len(french_character)))
decoder_lstm = LSTM(latent_dim,return_sequences=True,dropout=0.2,return_state=True)
decoder_outputs_1, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h_1,state_c_1])
decoder_lstm_1 = LSTM(latent_dim, return_sequences=True,dropout=0.2,return_state=True)
decoder_outputs, _, _ = decoder_lstm_1(decoder_outputs_1, initial_state=[state_h,state_c])
decoder_dense = Dense(len(french_character), activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
model=Model([encoder_inputs, decoder_inputs], decoder_outputs)

# -----------
print("Training")
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
model.save("engtofrench.h5")
