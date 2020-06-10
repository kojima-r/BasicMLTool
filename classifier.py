import pickle
import os
import argparse
import random
import numpy as np
import warnings
import pickle
from multiprocessing import Pool

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import RFE, RFECV
    from sklearn import svm
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.utils import resample
import csv
import json
import functools

# this project
from util import load_data, NumPyArangeEncoder


def objective(train_x, train_y, valid_x, valid_y, trial):
    import xgboost
    param={
        'max_depth':trial.suggest_int('max_depth',1,30),
        'learning_rate': trial.suggest_uniform('learning_rate',0.0,1),
        'round_num': trial.suggest_int('round_num',1,30),
    }

    clf = xgboost.XGBClassifier(
        max_depth=param['max_depth'],
        learning_rate=param['learning_rate'],
        round_num=param['round_num'])
    clf.fit(train_x,train_y)
    return 1.0-clf.score(valid_x,valid_y)

def optimize(x, y):
    import optuna
    import xgboost
    train_x,valid_x,train_y,valid_y=sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    obj_f = functools.partial(objective, train_x, train_y, valid_x, valid_y)
    study = optuna.create_study()
    study.optimize(obj_f, n_trials=100)
    clf = xgboost.XGBClassifier(**study.best_params)
    return clf

#############################################################################
# 識別のためのモデルとグリッドサーチのためのパラメータを返す　　　　　　　　#
#############################################################################
def get_classifier_model(args):
    if args.model == "rf":
        param_grid = {
            "n_estimators": [10, 100, 1000],
            "max_features": ["auto"],
            "min_samples_split": [2],
            "max_depth": [None],
        }
        clf = RandomForestClassifier()
    elif args.model == "xgb":
        import xgboost
        param_grid = {
            "max_depth": [x for x in range(3, 10, 2)],
            "min_child_weight": [x for x in range(1, 6, 2)],
            "subsample": [0.95],
            "colsample_bytree": [1.0],
            "n_estimators": [1000],
        }
        clf = xgboost.XGBClassifier()
    elif args.model == "lgb":
        import lightgbm as lgb

        param_grid = {"n_estimators": [1000]}
        clf = lgb.LGBMClassifier()
    elif args.model == "svm":
        param_grid = {"C": np.linspace(0.0001, 10, num=args.trials)}
        clf = svm.SVC(kernel="linear", probability=True)
    elif args.model == "poly_svm":
        param_grid = {"C": np.linspace(0.0001, 10, num=args.trials)}
        clf = svm.SVC(kernel="poly", probability=True)
    elif args.model == "rbf_svm":
        param_grid = {
            "C": np.linspace(0.0001, 10, num=args.trials),
            "gamma": np.linspace(0.01, 100, num=args.trials),
        }
        clf = svm.SVC(kernel="rbf", probability=True)
    elif args.model == "lr":
        param_grid = {"C": np.linspace(0.0001, 10, num=args.trials)}
        clf = sklearn.linear_model.LogisticRegression()
    else:
        raise Exception("[ERROR] unknown model:", args.model)
    return clf, param_grid


#############################################################################
# 回帰のためのモデルとグリッドサーチのためのパラメータを返す　　　　　　　　#
#############################################################################
def get_regressor_model(args):
    if args.model == "rf":
        param_grid = {
            "n_estimators": [10, 100, 1000],
            "max_features": ["auto"],
            "min_samples_split": [2],
            "max_depth": [None],
        }
        clf = RandomForestRegressor(n_estimators=1000)
    elif args.model == "lgb":
        import lightgbm as lgb

        param_grid = {"n_estimators": [1000]}
        clf = lgb.LGBMRegressor()
    elif args.model == "svm":
        param_grid = {"C": np.linspace(0.0001, 10, num=args.trials)}
        clf = svm.SVR(kernel="linear")
    elif args.model == "rbf_svm":
        param_grid = {
            "C": np.linspace(0.0001, 10, num=args.trials),
            "gamma": np.linspace(0.01, 100, num=args.trials),
        }
        clf = svm.SVR(kernel="rbf")
    elif args.model == "poly_svm":
        param_grid = {"C": np.linspace(0.0001, 10, num=args.trials)}
        clf = svm.SVR(kernel="poly")
    elif args.model == "en":
        param_grid = {
            "alpha": np.linspace(0.0001, 10, num=args.trials),
            "l1_ratio": np.linspace(0.0001, 1.0, num=args.trials),
        }
        clf = sklearn.linear_model.ElasticNet()
    elif args.model == "br":
        param_grid = {
            "alpha_1": np.linspace(0.0001, 10, num=args.trials),
            "alpha_2": np.linspace(0.0001, 10, num=args.trials),
        }
        clf = sklearn.linear_model.BayesianRidge()
    else:
        raise Exception("[ERROR] unknown model:", args.model)
    return clf, param_grid

def evaluate_group(test_y, pred_y, prob_y, test_group, args, result={}):
    group_pred_y={}
    group_prob_y={}
    group_test_y={}
    for g, y_prob, y_pred, y_test in zip(test_group,prob_y,pred_y,test_y):
        if g not in group_test_y:
            group_pred_y[g]=[]
            group_prob_y[g]=[]
            group_test_y[g]=[]
        group_pred_y[g].append(y_pred)
        group_prob_y[g].append(y_prob)
        group_test_y[g].append(y_test)
    g_list=[]
    pred_y_list=[]
    test_y_list=[]
    for g in group_pred_y.keys():
        y=np.mean(group_prob_y[g],axis=0)
        print(y)
        yi=np.argmax(y)
        label=group_test_y[g][0]
        g_list.append(g)
        pred_y_list.append(yi)
        test_y_list.append(label)
    if args.task == "binary" or args.task == "multiclass":
        precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
            test_y_list, pred_y_list, average="macro"
        )
        result["group_agg_precision"] = precision
        result["group_agg_recall"] = precision
        result["group_agg_f1"] = f1
        conf = sklearn.metrics.confusion_matrix(test_y_list, pred_y_list)
        result["group_agg_confusion"] = conf
        accuracy = sklearn.metrics.accuracy_score(test_y_list, pred_y_list)
        result["group_agg_accuracy"] = accuracy
    else:
        raise Exception("[ERROR] unknown task:", args.task)
    return result


################################################
# 評価を行う　　　　　　　                  　 #
# test_y:テストデータの正答　　　　　　　　    #
# pred_y:予測結果　　　　　　　　              #
# prob_y:予測スコア　　　　　　　　            #
# result:評価結果を保存するためのディクショナリ#
################################################
def evaluate(test_y, pred_y, prob_y, args, result={}):

    mask=~np.isnan(test_y)
    test_y=test_y[mask]
    pred_y=pred_y[mask]
    if prob_y is not None:
        prob_y=prob_y[mask]

    if args.task == "binary":
        ## ２値分類
        auc = sklearn.metrics.roc_auc_score(test_y, prob_y[:, 1], average="macro")
        roc_curve = sklearn.metrics.roc_curve(test_y, prob_y[:, 1], pos_label=1)
        result["roc_curve"] = roc_curve
        result["auc"] = auc
        precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
            test_y, pred_y
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
            roc_curve = sklearn.metrics.roc_curve(
                test_y == i, prob_y[:, i], pos_label=1
            )
            result["roc_curve"].append(roc_curve)
            result["auc"].append(auc)
        precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
            test_y, pred_y, average="macro"
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


################################################
# cross-validation の1 fold 分の計算           #
################################################
def train_cv_one_fold(arg):
    g=None
    if len(arg)==6:
        # groupの情報がある場合
        x, y, h, one_kf, g, args = arg
    else:
        # groupの情報がない場合
        x, y, h, one_kf, args = arg
    pipeline=[]
    ##
    ## 学習用セットとテスト用セットに分ける
    ##
    train_idx, test_idx = one_kf
    if args.train_data_sample is not None:
        train_idx = np.random.choice(train_idx, args.train_data_sample, replace=False)
    train_x = np.copy(x[train_idx])
    train_y = y[train_idx]
    test_x = np.copy(x[test_idx])
    test_y = y[test_idx]
    test_g = g[test_idx] if g is not None else None
    ##
    ## 手法を選択
    ##
    if args.task == "regression":
        clf, param_grid = get_regressor_model(args)
    else:
        clf, param_grid = get_classifier_model(args)
    result = {}
    ##
    ## 特徴選択を行う
    ##
    selected_feature = None
    if args.feature_selection:
        ##
        ## 特徴選択を行い、選択された特徴で予測をする
        ##
        if args.num_features is not None:
            rfe = RFE(clf, args.num_features)
        else:
            rfe = RFECV(clf,cv=3)
        mask=~np.isnan(train_y)
        rfe = rfe.fit(train_x[mask,:], train_y[mask])
        """
        # feature selection による予測結果を保存する場合はコメントをはずす
        result["feature_selection_pred_y"] = rfe.predict(test_x)
        prob_y = rfe.predict_proba(test_x) if hasattr(clf, "predict_proba") else None
        result["feature_selection_prob_y"] = prob_y
        """
        ##
        ## 選択された特徴を保存する
        ##
        selected_feature = rfe.support_
        print("=== selected feature ===")
        if h is None:
            selected_feature_name = [ i for i, el in enumerate(selected_feature) if el == True]
            print(len(selected_feature_name), ":", selected_feature_name)
        else:
            selected_feature_name = [ attr for attr, el in zip(h, selected_feature) if el == True]
            print(len(selected_feature_name), ":", selected_feature_name)
            result["selected_feature_name"] = selected_feature_name
        result["selected_feature"] = selected_feature
        result["feature_name"] = selected_feature
        ##
        ## 学習・テストデータをこのfold中、選択された特徴のみにする
        ##
        train_x = rfe.transform(train_x)
        test_x = rfe.transform(test_x)
        pipeline.append(rfe)
    if h is not None:
        result["feature_name"] = h

    if args.grid_search:
        ##
        ## グリッドサーチでハイパーパラメータを選択する
        ## ハイパーパラメータを評価するため学習セットを、さらに、パラメータを決定する学習セットとハイパーパラメータを評価するためのバリデーションセットに分けてクロスバリデーションを行う
        ##
        grid_search = sklearn.model_selection.GridSearchCV(
            clf, param_grid, cv=args.param_search_splits
        )
        mask=~np.isnan(train_y)
        grid_search.fit(train_x[mask,:], train_y[mask])

        ##
        ## 最も良かったハイパーパラメータや結果を保存
        ##
        print("Best parameters: {}".format(grid_search.best_params_))
        print("Best cross-validation: {}".format(grid_search.best_score_))
        result.update(
            {
                "param": grid_search.best_params_,
                "best_score": grid_search.best_score_,
            }
        )
        """
        ## 最も良かったハイパーパラメータのモデルを用いてテストデータで評価を行い、保存する場合はコメントをはずす
        pred_y = grid_search.predict(test_x)
        prob_y = grid_search.predict_proba(test_x) if hasattr(grid_search, "predict_proba") else  None
        result["grid_search_pred_y"] = pred_y
        prob_y = rfe.predict_proba(test_x) if hasattr(clf, "predict_proba") else None
        result["grid_search_prob_y"] = prob_y
        """
        ##
        ## 最も良かったハイパーパラメータの識別器を保存
        ## （学習データ全体での再フィッティングはこの段階では行わない）
        ##
        clf = grid_search.best_estimator_
    if args.opt:
        clf=optimize(train_x, train_y)

    ##
    ## clf を学習データ全体で再学習する
    ##
    mask=~np.isnan(train_y)
    clf.fit(train_x[mask,:], train_y[mask])
    ##
    ## 予測器ごとに特有の結果を出力する
    ##
    # ベイズ回帰の予測標準偏差
    if isinstance(clf, sklearn.linear_model.BayesianRidge):
        pred_y, pred_y_std = clf.predict(test_x, return_std=True)
        result["pred_y_std"] = pred_y_std
    else:
        pred_y = clf.predict(test_x)

    # 特徴量の重要度
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        result["feature_importance"] = fi
        fi_str = ",".join(map(str, fi))
        print("feature_importance", len(fi), ":", fi_str)

    # ランダムフォレストの予測標準偏差
    if isinstance(clf, RandomForestRegressor):
        if args.fci:
            import forestci as fci
            unbiased_var = fci.random_forest_error(clf, train_x, test_x)
            result["test_y_std"] = np.sqrt(unbiased_var)

    ##
    ## 予測結果やインデックスの保存
    ##
    result["test_y"] = test_y
    result["test_idx"] = test_idx
    result["test_group"] = test_g
    result["pred_y"] = pred_y
    prob_y=None
    if hasattr(clf, "predict_proba"):
        prob_y=clf.predict_proba(test_x)
    result["prob_y"] = prob_y
    pipeline.append(clf)
    ##
    ## 評価
    ##
    #if test_g is not None:
    #    result=evaluate_group(test_y, pred_y, prob_y, test_g, args, result=result)
    result = evaluate(test_y, pred_y, prob_y, args, result)
    if "accuracy" in result:
        if args.task == "binary":
            print("Cross-validation test accuracy: %3f" % (result["accuracy"]))
            print("Cross-validation test AUC: %3f" % (result["auc"]))
        if args.task == "multiclass":
            for i,auc in enumerate(result["auc"]):
                print("Task %d Cross-validation test AUC: %3f" % (i,auc))
            acc=result["accuracy"]
            print("Cross-validation test accuracy: %3f" % (acc))
    else:
        print("Cross-validation r2: %3f" % (result["r2"]))

    return (result, pipeline)


##############################################
# --- 学習処理の全体                     --- #
##############################################
def run_train(args):
    all_result = {}
    model_result = []
    for filename in args.input_file:
        print("=================================")
        print("== Loading data ... ")
        print("=================================")
        option = {}
        if args.group is not None:
            option["group"] = args.group
        x, y, opt, h, index = load_data(
            filename,
            ans_col=args.answer,
            ignore_col=args.ignore,
            header=args.header,
            cat_col=args.categorical,
            option=option,
        )
        g = None
        if args.group is not None or "group" in opt:
            if "group_type" in opt:
                if opt["group_type"]!="int":
                    print("group remapping")
                    g = []
                    mapping_g = {}
                    for g_name in opt["group"]:
                        if g_name not in mapping_g:
                            mapping_g[g_name] = len(mapping_g)
                        g.append(mapping_g[g_name])
                    g = np.array(g, dtype=np.int32)
                else:
                    g = np.array(opt["group"], dtype=np.int32)
        if args.data_sample is not None:
            x, y, g = resample(x, y, g, n_samples=args.data_sample)
        ## 欠損値を補完(平均)
        m = np.nanmean(x, axis=0)
        if h is not None:
            h = np.array(h)[~np.isnan(m)]
        imr = SimpleImputer(missing_values=np.nan, strategy="mean")
        x = imr.fit_transform(x)
        print("x:", x.shape)
        print("y:", y.shape)
        ## 標準化
        sc = StandardScaler()
        x = sc.fit_transform(x)

        print("x:", x.shape)
        print("y:", y.shape)
        if g is not None:
            print("g:", g.shape)
            print("grouping enabled:", g.shape)
        ## データから２クラス問題か多クラス問題化を決めておく
        if args.task == "auto":
            if len(np.unique(y)) == 2:
                args.task = "binary"
            else:
                args.task = "multiclass"
        if args.task != "regression":
            y = y.astype(dtype=np.int64)

        ##
        ## cross-validation を並列化して行う
        ##
        print("=================================")
        print("== Starting cross-validation ... ")
        print("=================================")
        if g is not None:
            kf = sklearn.model_selection.GroupKFold(n_splits=args.splits)
            pool = Pool(processes=args.splits)
            results = pool.map(
                train_cv_one_fold, [(x, y, h, s, g, args) for s in kf.split(x, y, g)]
            )
        else:
            kf = sklearn.model_selection.KFold(n_splits=args.splits, shuffle=True)
            pool = Pool(processes=args.splits)
            results = pool.map(
                train_cv_one_fold, [(x, y, h, s, args) for s in kf.split(x)]
            )

        ##
        ## cross-validation の結果をまとめる
        ## ・各評価値の平均・標準偏差を計算する
        ##
        cv_result = {"cv": [r[0] for r in results]}
        model_result.append([r[1] for r in results])
        print("=================================")
        print("== Evaluation ... ")
        print("=================================")
        if args.task == "regression":
            score_names = ["r2", "mse"]
        else:
            score_names = ["accuracy", "f1", "precision", "recall", "auc"]
        for score_name in score_names:
            scores = [r[0][score_name] for r in results]
            test_mean = np.nanmean(np.asarray(scores))
            test_std = np.nanstd(np.asarray(scores))
            print(
                "Mean %10s on test set: %3f (standard deviation: %3s)"
                % (score_name, test_mean, test_std)
            )
            cv_result[score_name + "_mean"] = test_mean
            cv_result[score_name + "_std"] = test_std
        ##
        ## 全体の評価
        ##
        test_y = []
        pred_y = []
        for result in cv_result["cv"]:
            test_y.extend(result["test_y"])
            pred_y.extend(result["pred_y"])
        if args.task != "regression":
            conf = sklearn.metrics.confusion_matrix(test_y, pred_y)
            cv_result["confusion"] = conf
        cv_result["task"] = args.task
        cv_result["index"] = index
        ##
        ## 結果をディクショナリに保存して返値とする
        ##
        all_result[filename] = cv_result
    return all_result, model_result


############################################################
# --- mainの関数：コマンド実行時にはここが呼び出される --- #
############################################################
if __name__ == "__main__":
    ##
    ## コマンドラインのオプションの設定
    ##
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument(
        "--grid_search", default=False, help="enebled grid search", action="store_true"
    )
    parser.add_argument(
        "--feature_selection",
        default=False,
        help="enabled feature selection",
        action="store_true",
    )
    parser.add_argument(
        "--input_file",
        "-f",
        nargs="+",
        default=None,
        help="input filename (txt/tsv/csv)",
        type=str,
    )
    parser.add_argument(
        "--trials", default=3, help="Trials for hyperparameters random search", type=int
    )
    parser.add_argument(
        "--splits",
        "-s",
        default=5,
        help="number of splits for cross validation",
        type=int,
    )
    parser.add_argument(
        "--param_search_splits",
        "-p",
        default=3,
        help="number of splits for parameter search",
        type=int,
    )
    parser.add_argument(
        "--header", "-H", default=False, help="number of splits", action="store_true"
    )
    parser.add_argument(
        "--answer", "-A", help="column number of answer label", type=int
    )
    parser.add_argument(
        "--categorical",
        "-C",
        nargs="*",
        default=[],
        help="column numbers for categorical data",
        type=int,
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
    parser.add_argument("--output_result_csv", default=None, help="output: csv", type=str)
    parser.add_argument("--output_model", default=None, help="output: pickle", type=str)
    parser.add_argument("--seed", default=20, help="random seed", type=int)
    parser.add_argument(
        "--num_features", default=None, help="select features", type=int
    )
    parser.add_argument(
        "--fci", default=False, help="enabled forestci", action="store_true"
    )
    parser.add_argument("--data_sample", default=None, help="re-sample data", type=int)
    parser.add_argument("--train_data_sample", default=None, help="re-sample training data", type=int)
    parser.add_argument(
        "--group",
        "-g",
        nargs="+",
        default=None,
        help="column number of group",
        type=int,
    )
    parser.add_argument(
        "--opt", default=False, help="enabled optimization", action="store_true"
    )

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
    all_result, model_result = run_train(args)
    ##
    ## 結果を簡易に表示
    ##
    if args.task == "regression":
        score_names = ["r2", "mse"]
    else:
        score_names = ["accuracy", "auc"]
    print("=================================")
    print("== summary ... ")
    print("=================================")
    metrics_names = sorted(
        [m + "_mean" for m in score_names] + [m + "_std" for m in score_names]
    )
    print("\t".join(["filename"] + metrics_names))
    for key, o in all_result.items():
        arr = [key]
        for name in metrics_names:
            arr.append("%2.4f" % (o[name],))
        print("\t".join(arr))

    ##
    ## 結果をjson ファイルに保存
    ## 予測結果やcross-validationなどの細かい結果も保存される
    ##
    if args.output_json:
        print("[SAVE]", args.output_json)
        fp = open(args.output_json, "w")
        json.dump(all_result, fp, indent=4, cls=NumPyArangeEncoder)

    ##
    ## 結果をjson ファイルに保存
    ## 予測結果やcross-validationなどの細かい結果も保存される
    ##
    if args.output_result_csv:
        print("[SAVE]", args.output_result_csv)
        fp = open(args.output_result_csv, "w")
        fp.write("\t".join(["filename","index","group","fold","y","pred_y","prob_y"]))
        fp.write("\n")
        data=[]
        for filename,obj in all_result.items():
            for fold,o in enumerate(obj["cv"]):
                idx_list=o["test_idx"]
                for i,idx in enumerate(idx_list):
                    y=o["test_y"][i]
                    pred_y=o["pred_y"][i]
                    g=""
                    if "test_group" in o:
                        g=o["test_group"][i]
                    prob_y=""
                    if "prob_y" in o:
                        prob_y=o["prob_y"][i][pred_y]
                    arr=[filename,idx,g,fold,y,pred_y,prob_y]
                    data.append(arr)
        for v in sorted(data):
            arr=list(map(str,v))
            fp.write("\t".join(arr))
            fp.write("\n")


    ##
    ## 結果をcsv ファイルに保存
    ##
    if args.output_csv:
        print("[SAVE]", args.output_csv)
        fp = open(args.output_csv, "w")
        if args.task == "regression":
            score_names = ["r2", "mse"]
        else:
            score_names = ["accuracy", "f1", "precision", "recall", "auc"]
        metrics_names = sorted(
            [m + "_mean" for m in score_names] + [m + "_std" for m in score_names]
        )
        fp.write("\t".join(["filename"] + metrics_names))
        fp.write("\n")
        for key, o in all_result.items():
            arr = [key]
            for name in metrics_names:
                arr.append("%2.4f" % (o[name],))
            fp.write("\t".join(arr))
            fp.write("\n")
    ##
    ## 学習済みモデルをpickle ファイルに保存
    ##
    if args.output_model:
        print("[SAVE]", args.output_model)
        with open(args.output_model, "wb") as f:
            pickle.dump(model_result, f)
