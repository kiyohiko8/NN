import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import os
import cg


"""
###システム概要
- 入力した単語を元に既知情報を推定
- 全単語データから既知情報・未知情報に分類
"""

#全単語データの読み込み（リスト型）
#まだ作れていない
all_word_list = []


# モデルを読み込む
model = model_from_json(open('predict_model.json').read())

# 学習結果を読み込む
model.load_weights('predict_weights.h5')

model.summary();

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1), metrics=['accuracy'])


#ユーザデータの推定値出力
predictions = model.predict(x_test)

#既知・未知推定情報の取得
for i, word in enumerate(predictions):
	if predintions[i] > 0.8:
		know_list.append(all_word_list[i])
		


"""
#ユーザが知っていると思われる単語を表示	
print(know_list)
"""





