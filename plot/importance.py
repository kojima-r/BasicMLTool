# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import datetime
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import csv
import dateutil
import argparse



############################################################
# --- mainの関数：コマンド実行時にはここが呼び出される --- #
############################################################
if __name__ == '__main__':
	##
	## コマンドラインのオプションの設定
	##
	parser = argparse.ArgumentParser(description = "Classification")
	parser.add_argument("--input_file","-f",nargs='+',default=None,
		help = "input filename (json)", type = str)
	parser.add_argument("--trials",default=3,
		help = "Trials for hyperparameters random search", type = int)
	parser.add_argument("--splits","-s", default=5,
		help = "number of splits for cross validation", type = int)
	parser.add_argument("--param_search_splits","-p", default=3,
		help = "number of splits for parameter search", type = int)
	parser.add_argument('--header','-H',default=False,
		help = "number of splits", action='store_true')
	parser.add_argument('--answer','-A',
		help = "column number of answer label", type=int)
	parser.add_argument('--ignore','-I',nargs='*',default=[],
		help = "column numbers for ignored data", type=int)
	parser.add_argument("--model",default="rf",
		help = "method (rf/svm/rbf_svm/lr)", type = str)
	parser.add_argument("--task",default="auto",
		help = "task type (auto/binary/multiclass/regression)", type = str)
	parser.add_argument('--output_json',default=None,
		help = "output: json", type=str)
	parser.add_argument('--output_csv',default=None,
		help = "output: csv", type=str)
	parser.add_argument('--output_model',default=None,
		help = "output: pickle", type=str)
	parser.add_argument('--seed',default=20,
		help = "random seed", type=int)
	parser.add_argument('--num_features',default=None,
		help = "select features", type=int)
	parser.add_argument("--fci",default=False,
		help = "enabled forestci", action="store_true")
	parser.add_argument('--data_sample',default=None,
		help = "re-sample data", type=int)
	
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
	with open(args.input_file[0]) as fp:
		obj=json.load(fp)
	for k,v in obj.items():
		for cv_count,fold in enumerate(v["cv"]):
			x=[]
			y=[]
			xlabel=[]
			print("fold:",cv_count)
			for i,name in enumerate(fold["feature_name"]):
				f=fold["feature_importance"][i]
				print(name, f)
				x.append(i)
				y.append(f)
				xlabel.append(name)
			plt.bar(x,y)
			plt.xticks(x,xlabel,rotation=90, size='small')
			plt.savefig("cv%02d.png"%(cv_count,))
			plt.clf()

	
