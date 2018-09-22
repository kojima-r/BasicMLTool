# BasicMLTool

教育・簡易的な実験目的のsci-kitlearn 等を用いた機械学習手法の実装です。

## 環境

scikit-learn やnumpyのインストールされた環境であれば動くはずです。オススメはanaconda環境です。

## サンプルの動かし方

### サンプルデータをダウンロードして、前処理を行います。

Linux環境であれば`get_iris_sample.sh`スクリプトを以下のように実行すれば可能です。
```
 sh get_iris_sample.sh
``` 

Windowsの場合は、
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
からファイルをダウンロードしてこのプロジェクトのディレクトリに置きます。
コマンドからこのプロジェクトのディレクトリに移動して、以下を実行してください。
```
python preprocessing_sample_iris.py
```
特に何も表示されなければ成功です。

### 前処理により作られるファイル

前処理後のファイルはdataset.csvに保存されます。
内容を確認すると以下のようになっており、
0-3列目はアヤメに関する特徴量となっています（ここでは最初の列を0列目とします）
```
Sepal Length（がく片の長さ）, Sepal Width（がく片の幅）, Petal Length（花びらの長さ）, Petal Width（花びらの幅）
```

4列目はSetosa, Versicolor, Virginicaの3品種で前処理段階でそれぞれ0,1,2という整数値に変換されます。

ここでの扱う問題は、0-3列目の特徴量（説明変数）から4列目の目的変数を予測するタスクになります。

### 学習・評価の実行

例えば、以下のようなコマンドでこのタスクに手法を適用し、評価を行うことができます。

```
python classifier.py -f dataset.csv -A 4 --model rf --feature_selection --grid_search --output_json test.json --output_csv test.csv
```
それぞれオプションの意味は以下のようになっています。

* `-f dataset.csv`　はデータセットのファイルを指定しています。
* `-A 4`は予測ラベルの列の番号を指定してます。今回は4列目が目的変数なので4としています。
* `--model rf`は手法を指定します。rfはランダムフォレストです。指定できる手法に関しては後述します。
* `--feature_selection`特徴選択を行います。選択するための手法は recursive feature elimination になります。（http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html）
* `--grid_search`　ハイパーパラメータを決めるためにグリッドサーチを行います。
* `--output_json test.json`：test.jsonに結果をjsonファイル形式で保存します。
* `--output_csv test.csv`：結果のサマリをcsv形式で保存します。

## オプション
データや手法を変える際には指定するオプションを適切に変更する必要があります。
代表的なオプションに以下のオプションがあります。

* `--grid_search`：グリッドサーチを有効にします。
* `--feature_selection` ：特徴選択を有効にします
* `--input_file INPUT_FILE [INPUT_FILE ...], -f INPUT_FILE [INPUT_FILE ...]`
　：データセットファイルを指定します。(txt(タブ区切り)/tsv/csv)を入力にすることができます。拡張子から自動的に判断するため、適切な拡張子にする必要があります。
*  `--splits SPLITS, -s SPLITS` ：最終的に評価をする際のクロスバリデーションのfold数を指定します。
*  `--param_search_splits PARAM_SEARCH_SPLITS, -p PARAM_SEARCH_SPLITS`
  ：グリッドサーチでパラメータを探索する際のクロスバリデーションのfold数を指定します。
*  `--header, -H` ：入力するデータセットファイルの１行目にヘッダがあるかどうかを指定します。変数名などが１行目に書かれている場合はこれを指定してください。ヘッダが２行以上あるファイルには対応していないので適切に前処理する必要があります。
*  `--answer ANSWER, -A ANSWER`：入力するデータセットファイルで目的変数となる列を指定してください。整数値のみに対応しているため、適切に前処理を行い整数にする必要があります。
*  `--ignore [IGNORE [IGNORE ...]], -I [IGNORE [IGNORE ...]]`：入力するデータセットファイルで、本プログラムで使用しない列を指定できます。例えば、データIDや氏名といった変数は予測に利用すべきではないため、ここで指定する必要があります。また、日付等や文字列等の本プログラムで扱えない変数がある場合も指定する必要があります。
*  `--model MODEL` ：予測に使うモデルを使用します。以下のモデルが使用できます。
   * `rf`　：ランダムフォレスト（http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html）
   * `svm` ：サポートベクターマシンのリニアカーネル（http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html）
   * `rbf_svm` ：サポートベクターマシンのrbfカーネル（http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html）
     　このオプションと特徴選択--feature_selectionオプションを同時に使うことはできません。 
   * `lr` ：ロジスティック回帰（http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html）

*  `--output_json OUTPUT_JSON`：結果を保存するjsonファイルを指定します。後述するフォーマットで保存されます。より細かく結果を追跡して、解釈するためにはこれを利用する必要があります。
*  `--output_csv OUTPUT_CSV`：結果のサマリを保存するcsvファイルを指定します。

## output jsonのフォーマット

jsonフォーマットで、テキスト形式なのでテキストエディタでも読むことができます。pythonで読み込んで簡単に利用することができます。
例えば、以下のようなコマンドで読み込むことができます。
```
import json

f = open('test.json', 'r')
data = json.load(f)
print(data)
```

dataの中身は階層構造を持った以下のような構造になっています。

<>カッコは入力したデータによって異なる名前になります。""はその名前でアクセスすることができます

* <ファイル名>
  * 'accuracy_mean'：cross-validationの正答率の平均
  * 'accuracy_std'：cross-validationの正答率の標準偏差
  * 'f1_mean'：cross-validationのF値の平均
  * 'f1_std'：cross-validationのF値の標準偏差
  * 'precision_mean'：cross-validationの精度の平均
  * 'precision_std'：cross-validationの精度の標準偏差
  * 'recall_mean'：cross-validationの再現率の平均
  * 'recall_std'：cross-validationの再現率の標準偏差
  * 'auc_mean'：cross-validationのROC-AUCの平均
  * 'auc_std'：cross-validationのROC-AUCの標準偏差
  * 'confusion'　：２次元の配列で混同行列が入っています。（http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html）
  * 'cv'：cross-validationのそれぞれのフォールドでの以下の結果が入っているリスト

'cv'内のリストの要素には以下のデータが入っている
  * 'test_y'：テストデータの正解データ
  * 'pred_y'：テストデータの予測ラベル
  * 'prob_y'：テストデータの全ラベルに対するスコア（確率）：データの数 x ラベルの数の２次元配列、pred_y はこのデータのnp.argmax(prob_y,axis=1)とみなすことができる。
  * 'selected_feature'：特徴選択した場合は選択された特徴：各特徴を使用したかどうかのtrue/falseのリスト
  * 'param'：パラメータサーチをした場合は選択されたパラメータ
  * 'best_score'：パラメータサーチをした場合は選択されたパラメータを使ったときのスコア
  * 'roc_curve'：ROCカーブが保存されている（http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html）
  * 'auc'：このfoldのテストデータを用いて計算されたROC-AUC
  * 'precision'：このfoldのテストデータを用いて計算された精度
  * 'recall'：このfoldのテストデータを用いて計算された再現率
  * 'f1'：このfoldのテストデータを用いて計算されたF値
  * 'confusion'：このfoldのテストデータを用いて計算された混同行列
  * 'accuracy'：このfoldのテストデータを用いて計算された正答率
   
### output_jsonで出力したファイルを処理するプログラム例

#### cross-validation での各フォールドでの正答率を表示する例
```
import json

f = open('test.json', 'r')
data = json.load(f)

for filename,result in data.items():
  print("filename=",filename)
  for i,fold_result in enumerate(result['cv']):
    print("fold",i,": accuracy=",fold_result['accuracy'])
```

#### cross-validation での各フォールドでのROC Curveを表示する例
`example/plot_roc.py` を参照

