from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import numpy as np
import pandas as pd
import os
import re

"""
ImageNetで学習済みのVGG16モデルを使って入力画像のクラスを予測する
"""

# 学習済みのVGG16をロード
# 構造とともに学習済みの重みも読み込まれる
model = load_model('vg19.h5')
# model.summary()

df = pd.DataFrame(index=[], columns=["image name", "きれいな初日の出", "きれいでない初日の出", "その他"])

try:
    for filename in os.listdir("2017-01-01"):
        if re.search("jpg$", filename):
            # 引数で指定した画像ファイルを読み込む
            # サイズはVGG16のデフォルトである224x224にリサイズされる
            img = image.load_img("2017-01-01/" + filename, target_size=(64, 64))

            # 読み込んだPIL形式の画像をarrayに変換
            x = image.img_to_array(img)
            x = x / 255
            

            # 3次元テンソル（rows, cols, channels) を
            # 4次元テンソル (samples, rows, cols, channels) に変換
            # 入力画像は1枚なのでsamples=1でよい
            x = np.expand_dims(x, axis=0)

            # Top-5のクラスを予測する
            # VGG16の1000クラスはdecode_predictions()で文字列に変換される
            y_prob = model.predict(x)
            res_list = list(y_prob[0])
            res_list.insert(0, filename)

            df = df.append(pd.Series(res_list, index=df.columns), ignore_index=True)
    df.to_csv("result-2017-01-01.csv", encoding="utf_8_sig")
except KeyboardInterrupt:
    # Shift-JIS 形式の CSV ファイル (employee.sjis.csv) して出力
    df.to_csv("result.csv", encoding="utf_8_sig")

