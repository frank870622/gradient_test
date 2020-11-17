# 導入函式庫
import numpy as np  
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from keras import optimizers
from matplotlib import pyplot as plt

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()
learning_rate = 0.1
lr_decay = 1e-6
dict_01 = {}
dict_01_flag = 0
dict_02 = {}
dict_02_flag = 0
dict_03 = {}
dict_03_flag = 0
f = open('./training.txt')
training_line = f.readline()
all_training = []
all_output = []
while training_line:
  a = training_line.split(",")
  if a[1] not in dict_01:
      dict_01[a[1]] = dict_01_flag
      dict_01_flag += 1
      a[1] = dict_01[a[1]]
  else:
      a[1] = dict_01[a[1]]
  if a[2] not in dict_02:
      dict_02[a[2]] = dict_02_flag
      dict_02_flag += 1
      a[2] = dict_02[a[2]]
  else:
      a[2] = dict_02[a[2]]
  if a[3] not in dict_03:
      dict_03[a[3]] = dict_03_flag
      dict_03_flag += 1
      a[3] = dict_03[a[3]]
  else:
      a[3] = dict_03[a[3]]
  #print(a)
  if a[41] == 'normal\n' or a[41] == 'normal':
      a[41] = 1
  else:
      a[41] = 0
  for i in range(0, 42):
      a[i] = float(a[i])
  #all_training.append([a[0], a[1], a[4], a[5], a[22], a[23]])
  all_training.append(a[0:41])
  all_output.append(a[41])
  training_line = f.readline()
all_training = np.array(all_training)
all_output = np.array([all_output]).T
#print(all_training)
#print(all_output)

f = open('./test.txt')
test_line = f.readline()
all_test = []
all_answer = []
while test_line:
  a = test_line.split(",")
  if a[1] not in dict_01:
      dict_01[a[1]] = dict_01_flag
      dict_01_flag += 1
      a[1] = dict_01[a[1]]
  else:
      a[1] = dict_01[a[1]]
  if a[2] not in dict_02:
      dict_02[a[2]] = dict_02_flag
      dict_02_flag += 1
      a[2] = dict_02[a[2]]
  else:
      a[2] = dict_02[a[2]]
  if a[3] not in dict_03:
      dict_03[a[3]] = dict_03_flag
      dict_03_flag += 1
      a[3] = dict_03[a[3]]
  else:
      a[3] = dict_03[a[3]]
  if a[41] == 'normal\n' or a[41] == 'normal':
      a[41] = 1
  else:
      a[41] = 0
  for i in range(0, 42):
      a[i] = float(a[i])
  #all_test.append([a[0], a[1], a[4], a[5], a[22], a[23]])
  all_test.append(a[0:41])
  all_answer.append(a[41])
  test_line = f.readline()
all_test = np.array(all_test)
all_answer = np.array([all_answer]).T
# 建立簡單的線性執行的模型
model = []
model = Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
#model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 
model.add(Dense(units=10, input_dim=41, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, activation='relu'))
# Add output layer
#model.add(Dense(units=2, input_dim=2, kernel_initializer='normal', activation='softmax'))
model.add(Dense(units=2, activation='relu'))

model.summary()


# 編譯: 選擇損失函數、優化方法及成效衡量方式
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) 
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
#y_TrainOneHot = np_utils.to_categorical(y_train) 
#y_TestOneHot = np_utils.to_categorical(y_test) 
y_TrainOneHot = np_utils.to_categorical(all_output) 
y_TestOneHot = np_utils.to_categorical(all_answer) 
#print(y_TrainOneHot)
#print(len(y_TrainOneHot))
#print(len(y_TestOneHot))
# 將 training 的 input 資料轉為2維
#X_train_2D = X_train.reshape(60000, 28*28).astype('float32')  
#X_test_2D = X_test.reshape(10000, 28*28).astype('float32')



#x_Train_norm = X_train_2D/255
#x_Test_norm = X_test_2D/255
x_Train_norm = all_training
x_Test_norm = all_test
#print(len(x_Train_norm))
#print(len(x_Test_norm))
# 進行訓練, 訓練過程會存在 train_history 變數中
#print(x_Train_norm)
#print(all_output)
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=50, batch_size=500, verbose=2,shuffle=True)  

# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  
"""
# 預測(prediction)
X = x_Test_norm[0:10,:]
predictions = model.predict_classes(X)
# get prediction result
print(predictions)

# 模型結構存檔
from keras.models import model_from_json
json_string = model.to_json()
with open("model.config", "w") as text_file:
    text_file.write(json_string)

    
# 模型訓練結果存檔
#model.save_weights("model.weight")
"""
