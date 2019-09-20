import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
from keras.models import load_model

past_days=30
coming_days=10
num_periods=20

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df = df['Close']
df.head()
# print(df)

plt.figure(figsize = (15,10))
plt.plot(df, label='Company A')
plt.legend(loc='best')
plt.show()
array = df.values.reshape(df.shape[0],1)
# array[:5]

from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
array = scl.fit_transform(array)
# array[:5]
# print(array.shape)

#split in Train and Test
division = len(array) - num_periods*coming_days
# print(division)
array_test = array[division-past_days:]
array_train = array[:division]

# print(array_test.shape)
# print(array_train.shape)


#It takes the data and splits in input X and output Y, by spliting in  30 past days as input X 
#and 10 coming days as Y.
def processData(data, past_days, coming_days,jump=1):
    X,Y = [],[]
    for i in range(0,len(data) -past_days -coming_days +1, jump):
        X.append(data[i:(i+past_days)])
        Y.append(data[(i+past_days):(i+past_days+coming_days)])
    return np.array(X),np.array(Y)

X_test,y_test = processData(array_test,past_days,coming_days,coming_days)
y_test = np.array([list(a.ravel()) for a in y_test])

X,y = processData(array_train,past_days,coming_days)
y = np.array([list(a.ravel()) for a in y])

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)
print(y_train.shape)
print(y_validate.shape)
print(y_test.shape)

NUM_NEURONS_FirstLayer = 50
NUM_NEURONS_SecondLayer = 30
EPOCHS = 100


model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(past_days,1), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))
model.add(Dense(coming_days))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_validate,y_validate),shuffle=True,batch_size=2, verbose=2)

plt.figure(figsize = (15,10))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()

Xt = model.predict(X_test)
# print(Xt.shape)
# print(X_test.shape)

plt.figure(figsize = (15,10))

for i in range(0,len(Xt)):
    plt.plot([x + i*coming_days for x in range(len(Xt[i]))], scl.inverse_transform(Xt[i].reshape(-1,1)), color='r')
    
plt.plot(0, scl.inverse_transform(Xt[i].reshape(-1,1))[0], color='r', label='Prediction') 
    
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Target')
plt.legend(loc='best')
plt.show()


# the red line represents a 10 days prediction  based on the 30 past days, 20 red lines because we  test on 20 periods. 

division = len(array) - num_periods*coming_days

leftover = division%coming_days+1
# print(division)
# print(leftover)
array_test = array[division-past_days:]
array_train = array[leftover:division]
Xtrain,ytrain = processData(array_train,past_days,coming_days,coming_days)
Xtest,ytest = processData(array_test,past_days,coming_days,coming_days)

Xtrain = model.predict(Xtrain)
Xtrain = Xtrain.ravel()

Xtest = model.predict(Xtest)
Xtest = Xtest.ravel()

y = np.concatenate((ytrain, ytest), axis=0)
plt.figure(figsize = (15,10))
# print(len(X_train))
# print(len(X_test))
# Data in Train/Validation
plt.plot([x for x in range(past_days+leftover, len(Xtrain)+past_days+leftover)], scl.inverse_transform(Xtrain.reshape(-1,1)), color='r', label='Train')
# Data in Test
plt.plot([x for x in range(past_days +leftover+ len(Xtrain), len(Xtrain)+len(Xtest)+past_days+leftover)], scl.inverse_transform(Xtest.reshape(-1,1)), color='y', label='Test')

#Data used
plt.plot([x for x in range(past_days+leftover, past_days+leftover+len(Xtrain)+len(Xtest))], scl.inverse_transform(y.reshape(-1,1)), color='b', label='Target')


plt.legend(loc='best')
plt.show()


