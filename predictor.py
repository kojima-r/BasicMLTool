import pickle
import os
import argparse
import random
import numpy as np
import warnings
import pickle

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import RFE, RFECV
    from sklearn import svm
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.utils import resample
import csv
import json

# this project
from util import load_data, NumPyArangeEncoder

################################################
# 評価を行う　　　　　　　                  　 #
# test_y:テストデータの正答　　　　　　　　    #
# pred_y:予測結果　　　　　　　　              #
# prob_y:予測スコア　　　　　　　　            #
# result:評価結果を保存するためのディクショナリ#
################################################
def evaluate(test_y, pred_y, prob_y, args, result={}):
    if args.task == "binary":
        ## ２値分類
        auc = sklearn.metrics.roc_auc_score(test_y, prob_y[:, 1], average="macro")
        roc_curve = sklearn.metrics.roc_curve(test_y, prob_y[:, 1], pos_label=1)
        result["roc_curve"] = roc_curve
        result["auc"] = auc
        precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
            test_y, pred_y, pos_label=1, average="binary"
        )
        result["precision"] = precision
        result["recall"] = precision
        result["f1"] = f1
        conf = sklearn.metrics.confusion_matrix(test_y, pred_y)
        result["confusion"] = conf
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_y)
        result["accuracy"] = accuracy
    elif args.task == "multiclass":
        ## 多値分類
        result["roc_curve"] = []
        result["auc"] = []
        for i in range(prob_y.shape[1]):
            auc = sklearn.metrics.roc_auc_score(
                test_y == i, prob_y[:, i], average="macro"
            )
            roc_curve = sklearn.metrics.roc_curve(test_y == i, prob_y[:, i])
            result["roc_curve"].append(roc_curve)
            result["auc"].append(auc)
        precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
            test_y, pred_y, average="macro", pos_label=1
        )
        result["precision"] = precision
        result["recall"] = precision
        result["f1"] = f1
        conf = sklearn.metrics.confusion_matrix(test_y, pred_y)
        result["confusion"] = conf
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_y)
        result["accuracy"] = accuracy
    elif args.task == "regression":
        ## 回帰問題
        result["r2"] = sklearn.metrics.r2_score(test_y, pred_y)
        result["mse"] = sklearn.metrics.mean_squared_error(test_y, pred_y)
    else:
        raise Exception("[ERROR] unknown task:", args.task)
    return result


def run_predicition(args, clf):
    all_result = {}
    for filename in args.input_file:
        print("=================================")
        print("== Loading data ... ")
        print("=================================")
        x, y, g, h = load_data(
            filename, ans_col=args.answer, ignore_col=args.ignore, header=args.header
        )
        if args.data_sample is not None:
            x, y, g = resample(x, y, g, n_samples=args.data_sample)
        ## 欠損値を補完(平均)
        imr = Imputer(missing_values=np.nan, strategy="mean", axis=0)
        x = imr.fit_transform(x)
        ## 標準化
        sc = StandardScaler()
        x = sc.fit_transform(x)

        print("x:", x.shape)
        print("y:", y.shape)
        ## データから２クラス問題か多クラス問題化を決めておく
        if args.task == "auto":
            if len(np.unique(y)) == 2:
                args.task = "binary"
            else:
                args.task = "multiclass"
        if args.task != "regression":
            y = y.astype(dtype=np.int64)
        print(args.task)
        if type(clf) is list:
            for c in clf[:-1]:
                print(c)
                x = c.transform(x)
            clf=clf[-1]
            pred_y = clf.predict(x)
        else:
            pred_y = clf.predict(x)
        prob_y = None
        if hasattr(clf, "predict_proba"):
            prob_y = clf.predict_proba(x)
        result = evaluate(y, pred_y, prob_y, args)
        result["pred_y"] = pred_y
        result["prob_y"] = prob_y
        result["test_y"] = y
        # a
        ## 全体の評価
        ##
        print("=================================")
        print("== Evaluation ... ")
        print("=================================")
        if args.task == "regression":
            score_names = ["r2", "mse"]
        else:
            score_names = ["accuracy", "f1", "precision", "recall", "auc"]
        for score_name in score_names:
            # print("Mean %10s on test set: %3f" % (score_name,result[score_name]))
            print(score_name, result[score_name])
        ##
        if args.task != "regression":
            conf = sklearn.metrics.confusion_matrix(y, pred_y)
            result["confusion"] = conf
        all_result[filename] = result
    return all_result


############################################################
# --- mainの関数：コマンド実行時にはここが呼び出される --- #
############################################################
if __name__ == "__main__":
    ##
    ## コマンドラインのオプションの設定
    ##
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument("--load_model", default=None, help="output: pickle", type=str)
    parser.add_argument(
        "--input_file",
        "-f",
        nargs="+",
        default=None,
        help="input filename (txt/tsv/csv)",
        type=str,
    )
    parser.add_argument(
        "--header", "-H", default=False, help="number of splits", action="store_true"
    )
    parser.add_argument(
        "--answer", "-A", help="column number of answer label", type=int
    )
    parser.add_argument(
        "--ignore",
        "-I",
        nargs="*",
        default=[],
        help="column numbers for ignored data",
        type=int,
    )
    parser.add_argument(
        "--model", default="rf", help="method (rf/svm/rbf_svm/lr)", type=str
    )
    parser.add_argument(
        "--task",
        default="auto",
        help="task type (auto/binary/multiclass/regression)",
        type=str,
    )
    parser.add_argument("--output_json", default=None, help="output: json", type=str)
    parser.add_argument("--output_csv", default=None, help="output: csv", type=str)
    parser.add_argument("--seed", default=20, help="random seed", type=int)
    parser.add_argument("--data_sample", default=None, help="re-sample data", type=int)

    ##
    ## コマンドラインのオプションによる設定はargsに保存する
    ##
    args = parser.parse_args()
    ##
    ## 乱数初期化
    ##
    np.random.seed(args.seed)

    ##
    ## 学習開始
    ##
    with open(args.load_model, "rb") as f:
        clfs = pickle.load(f)
        clf = clfs[0][0]
        print(clf)
    all_result = run_predicition(args, clf)
    ##
    ## 結果をjson ファイルに保存
    ## 予測結果やcross-validationなどの細かい結果も保存される
    ##
    if args.output_json:
        print("[SAVE]", args.output_json)
        fp = open(args.output_json, "w")
        json.dump(all_result, fp, indent=4, cls=NumPyArangeEncoder)
    ##
    if args.task == "regression":
        score_names = ["r2", "mse"]
    else:
        score_names = ["accuracy", "f1", "precision", "recall", "auc"]
    if args.output_csv:
        print("[SAVE]", args.output_csv)
        fp = open(args.output_csv, "w")
        if args.task == "regression":
            score_names = ["r2", "mse"]
        else:
            score_names = ["accuracy", "f1", "precision", "recall", "auc"]
        metrics_names = sorted(score_names)
        fp.write("\t".join(["filename"] + metrics_names))
        fp.write("\n")
        for key, o in all_result.items():
            arr = [key]
            for name in metrics_names:
                arr.append("%2.4f" % (o[name],))
            fp.write("\t".join(arr))
            fp.write("\n")
    #
