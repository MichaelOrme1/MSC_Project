import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten

import matplotlib.pyplot as plt




Training = np.load("TrainingData.npy",allow_pickle=True)
Testing = np.load("TestingData.npy",allow_pickle=True)
validation = np.load("ValidationData.npy",allow_pickle=True)



Xtrain = Training[0]
ytrain = Training[1]

Xtrain = np.array([np.array(val) for val in Xtrain])#Fixes issues with numpy loading
    
Xtrain = np.array([val.reshape(13,13) for val in Xtrain])#Reshape

Xtrain = np.array([np.expand_dims(val,0) for val in Xtrain])#Add virtual batch to start


ytrain =  np.array([np.array(val) for val in ytrain])#Fixes issues with numpy loading

ytrain =  np.array([val.reshape(1,12) for val in ytrain])#Reshape to fit model

train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))#Make into dataset

Xtest= Testing[0]
ytest = Testing[1]

Xtest = np.array([np.array(val) for val in Xtest])#Fixes issues with numpy loading
    
Xtest = np.array([val.reshape(13,13) for val in Xtest])#Reshape

Xtest = np.array([np.expand_dims(val,0) for val in Xtest])#Add virtual batch to start


ytest =  np.array([np.array(val) for val in ytest])#Fixes issues with numpy loading

ytest =  np.array([val.reshape(1,12) for val in ytest])#Reshape to fit model

test_dataset = tf.data.Dataset.from_tensor_slices((Xtest, ytest))#Make into dataset

Xvalidation = validation[0]
yvalidation = validation[1]

Xvalidation = np.array([np.array(val) for val in Xvalidation])#Fixes issues with numpy loading
    
Xvalidation = np.array([val.reshape(13,13) for val in Xvalidation])#Reshape

Xvalidation = np.array([np.expand_dims(val,0) for val in Xvalidation])#Add virtual batch to start


yvalidation =  np.array([np.array(val) for val in yvalidation])#Fixes issues with numpy loading

yvalidation =  np.array([val.reshape(1,12) for val in yvalidation])#Reshape to fit model

validation_dataset = tf.data.Dataset.from_tensor_slices((Xvalidation, yvalidation))#Make into dataset






       

       
callback = tf.keras.callbacks.EarlyStopping(patience = 10)#Stop if validation accuracy goes down, prevents overfitting

model = Sequential()
#model.add(Dense(12, input_shape = (1,13,13), activation='relu'))
model.add(Flatten(input_shape = (13,13)))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(12, activation='sigmoid'))
model.summary()

 
model.compile(loss='BinaryCrossentropy', optimizer="adam", metrics=['accuracy'])
history=model.fit(train_dataset, epochs=150, batch_size=10,verbose=2,validation_data=(validation_dataset))
#loss, accuracy = model.evaluate(test_dataset)
#print('Accuracy: %.2f' % (accuracy*100),'Loss: %.2f' % (loss))
#dot_img_file = '/tmp/model_1.png'
model.summary()
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
tf.keras.models.save_model(model,'Keras_Models/sigmoid_test_long_IMG')


#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



