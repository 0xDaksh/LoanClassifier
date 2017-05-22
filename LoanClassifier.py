
# coding: utf-8

# In[1]:

# Loan Classifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# fix for memory allocation ERROR
def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


# In[48]:

# data processing
df = pd.read_csv('bank.csv')
X = df.iloc[:, 0:9].values
df1 = pd.read_csv('bank1.csv')
Y = df1.iloc[:, 0].values
# label encoding

le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 5] = le.fit_transform(X[:, 5])
X[:, 7] = le.fit_transform(X[:, 7])
Y = le.fit_transform(Y)
# One Hot Encoding
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()

# Creating sets
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=1)

# feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# DATA PROCESSING ENDS


# In[49]:

# importing Deep Learning Libraries
from keras.models import Sequential
from keras.layers import Dense


# In[54]:

# creating the model
clf = Sequential([
    Dense(units=11, kernel_initializer='uniform', activation='relu', input_dim=19),
    Dense(units=11, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])


# In[55]:

# compiling classifier
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[56]:

# training model
clf.fit(X_train, Y_train, batch_size=10, epochs=100)


# In[57]:

# predicting the values
Y_pred = clf.predict(X_test)
Y_pred = (Y_pred > 0.5)


# In[ ]:

print(Y_pred)


# In[60]:

c = confusion_matrix(Y_test, Y_pred)
print(c)

