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
			valid_line=True
			if col_num==0:
				col_num=len(row)
			for i in range(col_num):
				if i in ignore_col:
					pass
				elif i == ans_col:
					try:
						y_vec.append(float(row[i]))
					except:
						valid_line=False
						print("[SKIP] could not convert string to float:",row[i])
						break
				else:
					if row[i]=="":
						x_vec.append(np.nan)
					else:
						x_vec.append(float(row[i]))
			if valid_line:
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

def extract_data(filename,save_filename,support,header=False,ignore_col=[],ans_col=[]):
	_,ext_in=os.path.splitext(filename)
	_,ext_out=os.path.splitext(save_filename)
	if ext_in==".csv":
		sep_in=","
	elif ext_in==".tsv":
		sep_in="\t"
	elif ext_in==".txt":
		sep_in="\t"
	else:
		print("[ERROR] unknown file format")
	if ext_out==".csv":
		sep_out=","
	elif ext_out==".tsv":
		sep_out="\t"
	elif ext_out==".txt":
		sep_out="\t"
	else:
		print("[ERROR] unknown file format")
	extract_data_xsv(filename,save_filename,support,header,ignore_col,ans_col,sep_in,sep_out)


def extract_data_xsv(filename,save_filename,support,header,ignore_col,ans_col,sep_in,sep_out):
	col_num=0
	ofp=open(save_filename,"w")
	with open(filename) as fp:
		tsv = csv.reader(fp, delimiter = sep_in)
		if header:
			row=next(tsv)
			if col_num==0:
				col_num=len(row)
			line=[]
			line_count=0
			for i in range(col_num):
				if i in ignore_col:
					line.append(row[i])
				elif i == ans_col:
					line.append(row[i])
				else:
					if support[line_count]:
						line.append(row[i])
					line_count+=1
			ofp.write(sep_out.join(line))
			ofp.write("\n")
		for row in tsv:
			if col_num==0:
				col_num=len(row)
			line=[]
			line_count=0
			for i in range(col_num):
				if i in ignore_col:
					line.append(row[i])
				elif i == ans_col:
					line.append(row[i])
				else:
					if support[line_count]:
						line.append(row[i])
					line_count+=1
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
			return obj.tolist() # or map(int, obj)
		return json.JSONEncoder.default(self, obj)


