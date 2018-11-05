import os
import argparse
import random
import numpy as np
import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	import sklearn
	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import Imputer
	#from sklearn.impute import SimpleImputer
	from sklearn.feature_selection import SelectKBest, f_regression
import csv
import json



# this project
from util import load_data,NumPyArangeEncoder,extract_data

def run_reduction(args):
	for file_counter,filename in enumerate(args.input_file):
		print("=================================")
		print("== Loading data ... ")
		print("=================================")
		x,y,h=load_data(filename,
			ans_col=args.answer,ignore_col=args.ignore,header=args.header)
		save_filename=None
		if args.output is not None:
			save_filename=args.output[file_counter]
		## 全て欠損の特徴を除去する
		a=np.all(np.isnan(x),axis=0)
		ig_index=np.where(a)[0]
		x=x[:,np.logical_not(a)]
		args.ignore.extend(ig_index)
		## 欠損値を補完(平均)
		if args.imputer:
			imr = Imputer(missing_values=np.nan, strategy='mean', axis=0,verbose=True)
			x = imr.fit_transform(x)
		## 標準化
		if args.std:
			sc = StandardScaler()
			x = sc.fit_transform(x)
		
		print("x:",x.shape)
		print("y:",y.shape)
		## データから２クラス問題か多クラス問題化を決めておく
		if args.task=="auto":
			if len(np.unique(y))==2:
				args.task="binary"
			else:
				args.task="multiclass"
		
	
		# 5つの特徴量を選択
		selector = SelectKBest(score_func=f_regression, k=100) 
		selector.fit(x, y)
		mask = selector.get_support()
		print(h)
		print(mask)
		print("====")
		new_h=[]
		for i,m in enumerate(mask):
			if m:
				new_h.append(h[i])
		print(new_h)
		
		if save_filename is not None:
			extract_data(filename,save_filename,mask,
				ans_col=args.answer,ignore_col=args.ignore,header=args.header)

		# 選択した特徴量の列のみ取得
		#X_selected = selector.transform(X)
		#print("X.shape={}, X_selected.shape={}".format(X.shape, X_selected.shape))
		
############################################################
# --- mainの関数：コマンド実行時にはここが呼び出される --- #
############################################################
if __name__ == '__main__':
	##
	## コマンドラインのオプションの設定
	##
	parser = argparse.ArgumentParser(description = "Classification")
	parser.add_argument("--input_file","-f",nargs='+',default=None,
		help = "input filename (txt/tsv/csv)", type = str)
	parser.add_argument("--trials",default=3,
		help = "Trials for hyperparameters random search", type = int)
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
	parser.add_argument('--output',default=None,nargs='+',
		help = "output: csv", type=str)
	parser.add_argument('--imputer',action='store_true',
		help = "enable imputer")
	parser.add_argument('--std',action='store_true',
		help = "enable standardization")
	parser.add_argument('--num_feature',default=100,type=int,
		help = "enable standardization")
	
	##
	## コマンドラインのオプションによる設定はargsに保存する
	##
	args = parser.parse_args()
	##
	## 乱数初期化
	##
	np.random.seed(20) 
	run_reduction(args)



