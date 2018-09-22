import pickle
import os
import argparse
import random
import warnings
with warnings.catch_warnings():
	import numpy as np
	import  sklearn
import csv
import json


def load_data_xsv(filename,header,ignore_col,ans_col,sep):
	x=[]
	y=[]
	hx=None
	hy=None
	col_num=0
	with open(filename) as fp:
		tsv = csv.reader(fp, delimiter = sep)
		if header:
			row=next(tsv)
			hx=[]
			hy=[]
			if col_num==0:
				col_num=len(row)
			for i in range(col_num):
				if i in ignore_col:
					pass
				elif i == ans_col:
					hy.append(row[i])
				else:
					hx.append(row[i])
		for row in tsv:
			x_vec=[]
			y_vec=[]
			if col_num==0:
				col_num=len(row)
			for i in range(col_num):
				if i in ignore_col:
					pass
				elif i == ans_col:
					y_vec.append(int(row[i]))
				else:
					x_vec.append(float(row[i]))
			x.append(x_vec)
			y.append(y_vec[0])
	return np.array(x),np.array(y),hx

def load_data(filename,header=False,ignore_col=[],ans_col=[]):
	_,ext=os.path.splitext(filename)
	if ext==".csv":
		return load_data_xsv(filename,header,ignore_col,ans_col,",")
	elif ext==".tsv":
		return load_data_xsv(filename,header,ignore_col,ans_col,"\t")
	elif ext==".txt":
		return load_data_xsv(filename,header,ignore_col,ans_col,"\t")
	else:
		print("[ERROR] unknown file format")
	return None,None,None

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
			return obj.tolist() # or map(int, obj)
		return json.JSONEncoder.default(self, obj)


