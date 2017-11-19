import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import os
import cg


"""
ここでデータセットのインポート

# 次元の指定
in_num =
hidden_1 =
hidden_2 =  
:
:
out_num = 


epoch_num = 
batch_size = 
"""






#データのを引っ張ってくる

x_data = 
y_data = 


#訓練データ、テストデータの作成
x_train, x_test, y_train, y_test =\
	train_test_split(x_data, y_data, train_size = 0.8)
	
#検証データの作成
x_train, x_val, y_train, y_val =\
	train_test_split(x_train, y_train, train_size = 0.9)



"""モデル構築"""

#パラメータの設定
epoch_num  = 50
batch_size = 100
in_num   = len(x_train[0])
hidden_1 = 100
hidden_2 = 100
out_num  = len(y_train[0])


#モデル構築,学習
model = Sequential()
model.add(Dense(input_dim = in_num, output_dim = hidden_1))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(output_dim = out_num))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1), metrics=['accuracy'])
history = model.fit(x_train, y_train, nb_epoch = epoch_num, batch_size=batch_size,validation_data=(x_val, y_val))


#モデル評価
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#テストデータと結果の表示
predictions = model.predict(x_test)
correct = y_test[:, np.newaxis]
print("x_test:")
print(x_test[0])
print("predictions:")
print(predictions[0])
print("correct:")
print(correct[0])



#モデルの保存
print("Saving Model...")
json_string = model.to_json()
open('predict_model.json', 'w').write(json_string)
print("Saved!")

#パラメータの保存
print("Saving Param...")
model.save_weights('predict_weights.h5')

gc.collect()



	
