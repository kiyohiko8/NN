import os, sys
import numpy as np
import math



"""モデルの構築"""

encoding_dim = 100
input_tw = Input(shape=(200,))#入力層
encoded = Dense(encoding_dim, activation = "relu")(input_tw) #中間層：特徴を抽出
decoded = Dense(200, activation = "sigmoid")(encoded)#出力層

autoencoder = Model(input=input_tw, output=decoded)

autoencoder.compile(optimizer="adadelta", loss='binary_crossentropy')




"""入力情報"""

#訓練データの生成
x_train = np.sin(np.radians(np.random.randint(0, 360, (100000, 200))))
#テストデータの生成
x_test = np.sin(np.radians(np.random.randint(0, 360, (100000, 200))))
	
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))



"""
重みデータの読み書き
autoencoder.save_weights('autoencoder.h5')
autoencoder.load_weights('autoencoder.h5')
"""



#実行

if __name__ == "__main__":
	autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=256, shuffle=True, validation_data= (x_test, x_test))
	
