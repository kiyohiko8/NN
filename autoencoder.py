"""
オートエンコーダモデル
kerasを使用
"""

from keras.layers import Input, Dense
from keras.models import Model 
import os, sys
import numpy as np



"""pathの指定"""
os.chdir("'作業ディレクトリ名'")




"""モデルの構築"""

encoding_dim = 100　#中間層の次元数
input_tw = Input(shape=(200,))#入力層
encoded = Dense(encoding_dim, activation = "relu")(input_tw) #中間層：特徴を抽出
decoded = Dense(200, activation = "sigmoid")(encoded)#出力層

#モデル化
autoencoder = Model(input=input_tw, output=decoded)

autoencoder.compile(optimizer="adadelta", loss='binary_crossentropy')




"""入力情報"""
#データの読み込み

#訓練データ


"""
重みデータの読み書き
#重みパラメータの保存
autoencoder.save_weights('autoencoder.h5')
#重みパラメータの呼び出し
autoencoder.load_weights('autoencoder.h5')
"""


#実行
if __name__ == "__main__":
	autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=256, suffle=True, validation_data= (x_test, x_test)
