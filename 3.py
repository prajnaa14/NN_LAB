import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

iris=load_iris()
x,y=iris.data,iris.target
y=to_categorical(y)
scalar=StandardScaler()
x=scalar.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

model=Sequential([
    Dense(64,activation='relu',input_shape=(x.shape[1],)),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,verbose=1,batch_size=8,epochs=10)

loss,accuracy=model.evaluate(x_test,y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")