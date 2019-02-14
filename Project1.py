from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Dense,Activation,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import h5py
import numpy
import cv2

#batch size
b=32

#training dataset
train_gen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip='true',fill_mode='nearest')
train = train_gen.flow_from_directory('data/train',target_size=(150,150),batch_size=b,class_mode='categorical')

#test dataset
test_gen = ImageDataGenerator(rescale=1./255)
test = test_gen.flow_from_directory('data/test',target_size=(150,150),batch_size=b,class_mode='categorical')

#Network
model = Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

#compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#fitting the model
model.fit_generator(train,epochs=20,verbose=2,callbacks=[EarlyStopping(monitor='acc',patience=2)])

#evaluating the model
a = model.evaluate_generator(test,verbose=0)
print('Accuracy :'+str(a[1]))

#saving the model
model.save('Project1.h5')

#load model
model = load_model('Project1.h5')

#image for prediction
img = cv2.imread('00001.jpg')
img = cv2.resize(img,(150,150))
img = numpy.reshape(img,(1,150,150,3))

p_gen = ImageDataGenerator(rescale=1./255)
pred = p_gen.flow(img)
p = model.predict_generator(pred,verbose=0)
p_class = p.argmax()

#cv2.imshow('window',img)
classes = ['Bus','Car','Truck']
print('Predicted class is',classes[p_class])

