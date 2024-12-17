from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

x,y=make_moons(n_samples=500)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)

activations=['sigmoid','tanh','relu','softmax']
histories={}

for act in activations:
    model=Sequential([
        Dense(16,activation=act,input_dim=2),
        Dense(8,activation=act),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    histories[act]=model.fit(x_train,y_train, validation_data=(x_test,y_test),verbose=0,epochs=20).history

    plt.plot(histories[act]['val_accuracy'],label=act)
plt.title("validation accuracy")
plt.legend()
plt.show()