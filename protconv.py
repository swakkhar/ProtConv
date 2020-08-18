
pip install tape_proteins

import torch

import numpy as np


"""# First Read files"""

def readFasta(filename):
    f=open(filename,'r')
    labels=[]
    seqs=[]
    i=0
    for line in f.readlines():
        if i%2==0:
            sym=line.rstrip().split("|")[1]
            labels.append(float(sym))
        else:
            seqs.append(line.rstrip())
        i=i+1
    return labels,seqs
train_labels,train_seqs=readFasta("/content/gdrive/My Drive/ACP/train.txt")  # change to your own path
test_labels,test_seqs=readFasta("/content/gdrive/My Drive/ACP/test.txt") # change to your own path

def readFasta(filename1,filename2):
    f1=open(filename1,'r')
    labels=[]
    seqs=[]
    i=0
    for line in f1.readlines():
        if i%2==0:
            labels.append(1)
        else:
            seqs.append(line.rstrip())
        i=i+1
    f1.close()

    f2=open(filename2,'r')
    i=0
    for line in f2.readlines():
        if i%2==0:
            labels.append(0)
        else:
            seqs.append(line.rstrip())
        i=i+1
    f2.close()
    return labels,seqs
train_labels,train_seqs=readFasta("/content/gdrive/My Drive/ProInFuse/train_pos.txt","/content/gdrive/My Drive/ProInFuse/train_neg.txt") # change to your own path
test_labels,test_seqs=readFasta("/content/gdrive/My Drive/ProInFuse/ind_pos.txt","/content/gdrive/My Drive/ProInFuse/ind_neg.txt") # change to your own path

# use any of the read files


num_of_features = 768
import numpy as np
X=np.zeros((len(train_seqs),num_of_features))
y=np.zeros(len(train_seqs))

ind_X=np.zeros((len(test_seqs),num_of_features))
ind_y=np.zeros(len(test_seqs))

# now lets populate X
i=0
for s in train_seqs:
    #f=extractFeatures(s)
    token_ids = torch.tensor([tokenizer.encode(s)])
    output = model(token_ids)
    sequence_output = output[0]
    pooled_output = output[1]
    
    X[i]=np.array(np.mean(sequence_output.detach().numpy(),axis=1) )
    i=i+1
    
y=np.array(train_labels)


i=0
for s in test_seqs:
    #f=extractFeatures(s)
    token_ids = torch.tensor([tokenizer.encode(s)])
    output = model(token_ids)
    sequence_output = output[0]
    pooled_output = output[1]
    ind_X[i]=np.array(np.mean(sequence_output.detach().numpy(),axis=1) )
    i=i+1

ind_y=np.array(test_labels)

print(X.shape)
print(ind_X.shape)



X_padded = np.concatenate((X, np.zeros((X.shape[0],16))), axis=1)
ind_X_padded = np.concatenate((ind_X, np.zeros((ind_X.shape[0],16))), axis=1)



from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X_padded)
X_normalized = std_scale.transform(X_padded)
ind_X_normalized  = std_scale.transform(ind_X_padded)


shape_unit=28
X_images=np.reshape(X_normalized,(X_normalized.shape[0],shape_unit,shape_unit))
ind_X_images = np.reshape(ind_X_normalized,(ind_X_normalized.shape[0],shape_unit,shape_unit))

print(X_images.shape)
print(ind_X_images.shape)

# %tensorflow_version 1.x
#import all libraries
#here comes the deep learning part
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D, Dense, Flatten, Dropout,MaxPooling2D
from keras import layers
from keras.layers.pooling import MaxPooling1D,MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras import regularizers


model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(3, 3), padding="same",activation='relu', input_shape=(28,28,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same",activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3),padding="same", activation='relu'))
model.add(layers.AveragePooling2D())


model.add(layers.Flatten())

model.add(layers.Dense(units=128, activation='relu',kernel_regularizer=regularizers.l2(1e-4),bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))

model.add(layers.Dense(units=64, activation='relu',kernel_regularizer=regularizers.l2(1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)))

model.add(layers.Dense(units=1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

eX = np.expand_dims(all_images, -1)
eY   = np.expand_dims(all_y, -1)
eIX = np.expand_dims(ind_X_images, -1)
eIY   = np.expand_dims(ind_y, -1)

history = model.fit(x=eX,y=eY,epochs=50,shuffle=True,validation_data=(eIX, eIY),verbose=1,batch_size=64)

# Plot history: MAE
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='training data')
plt.plot(history.history['val_accuracy'], label='validation data')
#plt.title('')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()