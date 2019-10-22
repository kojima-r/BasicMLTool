import pickle
import os
import argparse
import random
import warnings

with warnings.catch_warnings():
    import numpy as np
    import sklearn
import csv
import json


def onehot(x, hx, cat_col, category):
    new_x = []
    new_hx = []
    prev_c = None
    for c in cat_col:
        if prev_c is None:
            tmp = x[:, :c]
        else:
            tmp = x[:, prev_c + 1 : c]

        prev_c = c
        tmp = x[:, c + 1 : c]
    return x, y, hx


def flatten(l):
    ret = []
    for a in l:
        if hasattr(a, "__iter__"):
            ret += flatten(a)
        else:
            ret.append(a)
    return ret


def load_data_xsv(filename, header, ignore_col, ans_col, sep, cat_col=[], option={}):
    x = []
    y = []
    hx = None
    hy = None
    col_num = 0
    categorical = {}
    option_data = {k: [] for k in option.keys()}
    option_vals = flatten(option.values())
    with open(filename) as fp:
        tsv = csv.reader(fp, delimiter=sep, quotechar='"')
        if header:
            row = next(tsv)
            hx = []
            hy = []
            if col_num == 0:
                col_num = len(row)
                print("the number of columns=", col_num)
            for i in range(col_num):
                if i in ignore_col:
                    pass
                elif i in option_vals:
                    pass
                elif i == ans_col:
                    hy.append(row[i])
                else:
                    hx.append(row[i])
        for line_no, row in enumerate(tsv):
            x_vec = []
            y_vec = []
            valid_line = True
            option_flag = {k: True for k in option.keys()}
            if col_num == 0:
                col_num = len(row)
            for i in range(col_num):
                if i in ignore_col:
                    pass
                elif i == ans_col:
                    try:
                        y_vec.append(float(row[i]))
                    except:
                        valid_line = False
                        print("[SKIP] could not convert string to float:", row[i])
                        break
                elif i in option_vals:
                    for key, value in option.items():
                        if (hasattr(value, "__iter__") and i in value) or (i == value):
                            if option_flag[key]:
                                option_data[key].append(row[i])
                                option_flag[key] = False
                            else:
                                s = option_data[key][-1]
                                option_data[key][-1] = s + "-" + row[i]
                elif i in cat_col:
                    if row[i] not in categorical:
                        categorical[row[i]] = len(categorical)
                    x_vec.append(categorical[row[i]])
                else:
                    if i >= len(row):
                        x_vec.append(np.nan)
                        print(
                            "[WARN] Line",
                            line_no,
                            "is length",
                            len(row),
                            " and shorter than its expectation",
                            col_num,
                        )
                    elif row[i] == "":
                        x_vec.append(np.nan)
                    else:
                        x_vec.append(float(row[i]))
            if valid_line:
                x.append(x_vec)
                if len(y_vec) > 0:
                    y.append(y_vec[0])

    return np.array(x), np.array(y), option_data, hx


def load_data(filename, header=False, ignore_col=[], ans_col=[], cat_col=[], option={}):
    print(filename)
    if "," in filename:
        pair = filename.split(",")
        print("[LOAD]", pair[0])
        print("[LOAD]", pair[1])
        print("[LOAD]", pair[2])
        x = np.load(pair[0])
        y = np.load(pair[1])
        opt = {}
        opt["group"] = np.load(pair[2])

        return x, y, opt, None
    _, ext = os.path.splitext(filename)
    if ext == ".csv":
        return load_data_xsv(
            filename, header, ignore_col, ans_col, ",", cat_col, option
        )
    elif ext == ".tsv":
        return load_data_xsv(
            filename, header, ignore_col, ans_col, "\t", cat_col, option
        )
    elif ext == ".txt":
        return load_data_xsv(
            filename, header, ignore_col, ans_col, "\t", cat_col, option
        )
    else:
        print("[ERROR] unknown file format")
    return None, None, None, None


def extract_data(
    filename, save_filename, support, header=False, ignore_col=[], ans_col=[]
):
    _, ext_in = os.path.splitext(filename)
    _, ext_out = os.path.splitext(save_filename)
    if ext_in == ".csv":
        sep_in = ","
    elif ext_in == ".tsv":
        sep_in = "\t"
    elif ext_in == ".txt":
        sep_in = "\t"
    else:
        print("[ERROR] unknown file format")
    if ext_out == ".csv":
        sep_out = ","
    elif ext_out == ".tsv":
        sep_out = "\t"
    elif ext_out == ".txt":
        sep_out = "\t"
    else:
        print("[ERROR] unknown file format")
    extract_data_xsv(
        filename, save_filename, support, header, ignore_col, ans_col, sep_in, sep_out
    )


def extract_data_xsv(
    filename, save_filename, support, header, ignore_col, ans_col, sep_in, sep_out
):
    col_num = 0
    ofp = open(save_filename, "w")
    with open(filename) as fp:
        tsv = csv.reader(fp, delimiter=sep_in)
        if header:
            row = next(tsv)
            if col_num == 0:
                col_num = len(row)
            line = []
            line_count = 0
            for i in range(col_num):
                if i in ignore_col:
                    line.append(row[i])
                elif i == ans_col:
                    line.append(row[i])
                else:
                    if support[line_count]:
                        line.append(row[i])
                    line_count += 1
            ofp.write(sep_out.join(line))
            ofp.write("\n")
        for row in tsv:
            if col_num == 0:
                col_num = len(row)
            line = []
            line_count = 0
            for i in range(col_num):
                if i in ignore_col:
                    line.append(row[i])
                elif i == ans_col:
                    line.append(row[i])
                else:
                    if support[line_count]:
                        line.append(row[i])
                    line_count += 1
            ofp.write(sep_out.join(line))
            ofp.write("\n")
    return


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)
