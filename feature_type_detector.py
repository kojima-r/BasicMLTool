import os
import argparse
import random
import pandas as pd
import numpy as np
import warnings
import goodtables

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
import csv
import json

import collections
# this project
from util import load_data, NumPyArangeEncoder, extract_data

def detect_header(row):
    c=[classify_cell_type(cell) for cell in row]
    counter=collections.Counter(c)
    if counter["string"]/len(row)>0.8:
        return True
    else:
        return False
def detect_type(counter):
    n=sum([v for k,v in counter.items()])
    missing_cnt=0
    if "missing" in counter:
        missing_cnt=counter["missing"]
        r=counter["missing"]/n
        if r>0.5:
            return "missing"
    number_type=None
    number_cnt=0
    if "float" in counter:
        number_type="float"
        number_cnt+=counter["float"]
        if "integer" in counter:
            number_cnt+=counter["integer"]
    elif "integer" in counter:
        number_type="integer"
        number_cnt=counter["integer"]
    # vs string
    string_cnt=0
    if "string" in counter:
        string_cnt+=counter["string"]
    if number_cnt >= string_cnt+missing_cnt:
        return number_type
    if number_cnt + string_cnt >= missing_cnt:
        return "string"
    return "missing"

def classify_cell_type(cell):
    if cell=="":
        return "missing"
    else:
        try:
            int(cell)
            return "integer"
        except:
            pass
        try:
            float(cell)
            return "float"
        except:
            pass
        return "string"

def auto_detect(filename,header="auto",max_length=1000):
    _, ext = os.path.splitext(filename)
    if ext == ".csv":
        sep=","
    elif ext == ".tsv" or ext == ".txt":
        sep="\t"
    else:
        print("[ERROR] unknown file format")

    with open(filename) as fp:
        tsv = csv.reader(fp, delimiter=sep, quotechar='"')
        col_num=None
        col_data=None
        col_datatype=None
        col_name=None
        for line_no, row in enumerate(tsv):
            if line_no>=max_length:
                break
            if col_num is None:
                col_num=len(row)
                col_data=[[] for _ in range(col_num)]
                col_datatype=[[] for _ in range(col_num)]
                if header=="auto":
                    header_enabled=detect_header(row)
                else:
                    header_enabled=header
                if header_enabled:
                    col_name=row
                    continue
            elif col_num!=len(row):
                print("[ERROR] mismatch col num: line ",line_no)
            for j,cell in enumerate(row):
                col_data[j].append(cell)
                col_datatype[j].append(classify_cell_type(cell))
    print("Header:",header_enabled)
    enable_cnt=0
    col_type_list=[]
    for i,c in enumerate(col_datatype):
        counter=collections.Counter(c)
        col_type=detect_type(counter)
        col_type_list.append(col_type)
        if col_type!="missing":
            enable_cnt+=1
            if col_name is not None:
                print(i, col_name[i], ":",col_type)
            else:
                print(i,":",col_type)
            #for j,cnt in counter.most_common():
            #    print("  ",j,cnt/len(c))
    print(enable_cnt,"/",len(col_datatype))
    return header_enabled,col_type_list

def save_file(filename,out_filename,enable_col):
    _, ext = os.path.splitext(filename)
    if ext == ".csv":
        sep=","
    elif ext == ".tsv" or ext == ".txt":
        sep="\t"
    else:
        print("[ERROR] unknown file format")
    data=[]
    with open(filename) as fp:
        tsv = csv.reader(fp, delimiter=sep, quotechar='"')
        for row in tsv:
            new_row=[row[c] for c in enable_col]
            data.append(new_row)
    print("[SAVE]",out_filename)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f,delimiter=sep)
        writer.writerows(data)


############################################################
# --- mainの関数：コマンド実行時にはここが呼び出される --- #
############################################################
if __name__ == "__main__":
    ##
    ## コマンドラインのオプションの設定
    ##
    parser = argparse.ArgumentParser(description="Classification")
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
    parser.add_argument(
        "--output", default=None, nargs="+", help="output: csv", type=str
    )
    parser.add_argument("--imputer", action="store_true", help="enable imputer")
    parser.add_argument("--std", action="store_true", help="enable standardization")
    parser.add_argument(
        "--num_feature", default=100, type=int, help="enable standardization"
    )

    ##
    ## コマンドラインのオプションによる設定はargsに保存する
    ##
    args = parser.parse_args()
    ##
    ## 乱数初期化
    ##
    np.random.seed(20)
    for i,filename in enumerate(args.input_file):
        header_flag,col_datatype=auto_detect(filename)
        enable_col=[]
        for j,c in enumerate(col_datatype):
            if c!="missing":
                enable_col.append(j)
        #print(enable_col)
        if args.output:
            save_file(filename,args.output[i],enable_col)

    #run_attr_classifier(args)
