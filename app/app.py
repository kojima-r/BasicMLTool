# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for,make_response,jsonify
import numpy as np
import os
from datetime import datetime
import werkzeug
import subprocess
import glob
import json
import hashlib
import random, string

def randomname(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return ''.join(randlst)

def calculate_key(filename):
    text=(filename+randomname(5)).encode('utf-8')
    result = hashlib.md5(text).hexdigest()
    saveFileName = werkzeug.utils.secure_filename(result)
    return saveFileName


# 自身の名称を app という名前でインスタンス化する

template_dir = os.path.abspath('view')
app = Flask(__name__,template_folder=template_dir)
import data
app.register_blueprint(data.data1)
app.register_blueprint(data.data2)

worker={}
latest_setting={}

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/run', methods=['GET','POST'])
def post_run_init():
    model_name="rf"
    if request.method == 'POST':
        wid=request.form["wid"]
        if "model" in request.form:
            model_name=request.form["model"]
    else:
        wid=request.args.get('wid')
        if request.args.get('model') is not None:
            model_name=request.args.get('model')
    
    cmd=["python", "../classifier.py"]
    cmd+=["-f", "data/"+wid+".csv"]
    cmd+=["-H"]
    cmd+=["-A", "4"]
    cmd+=["--feature_selection"]
    cmd+=["--grid_search"]
    cmd+=["--output_json", "result/"+wid+".json"]
    cmd+=["--output_csv", "result/"+wid+".csv"]

    print(cmd)
    output_file=open('log/'+wid+'.txt', 'w')
    p = subprocess.Popen(cmd, stdout=output_file, text=True)
    worker[wid]={"process":p,"setting":[]}
    return make_response(jsonify({'worker_id':wid}))


UPLOAD_PDF_DIR="./data/"
@app.route('/upload', methods=['POST'])
def post_pdf_up():
    print(request.files)
    if 'files[]' in request.files:
        file = request.files['files[]']
        fileName = file.filename
        wid=calculate_key(fileName)
        file.save(os.path.join(UPLOAD_PDF_DIR, wid+".csv"))
        return make_response(jsonify({'result':wid}))

@app.route('/status/<wid>', methods=['GET'])
def status(wid=None):
    if wid not in worker or  worker[wid] is None:
        return make_response(jsonify({'worker_id':wid,'status':"not found"}))
    lines=[l for l in open("log/"+wid+".txt","r")]
    if worker[wid]["process"].poll() is None:
        return make_response(jsonify({'worker_id':wid,'status':"running","log":lines}))
    worker[wid]=None
    #result/4e9a9dc934fb9f46c0ee0f7e5b676f85.json
    obj = json.load(open("result/"+wid+".json","r"))
    return make_response(jsonify({'worker_id':wid,'status':"finished","log":lines,"result":obj}))

@app.route('/list/csv', methods=['GET'])
def list_pdf():
    l=glob.glob(UPLOAD_PDF_DIR+"*.csv")
    return make_response(jsonify(l))

MODEL_DIR="../ext/reader/model/"
@app.route('/list/model', methods=['GET'])
def list_model():
    l=[]
    for filename in glob.glob(MODEL_DIR+"*"):
        if os.path.isdir(filename):
            l.append(os.path.basename(filename))
    return make_response(jsonify(l))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0',port=5002) # どこからでもアクセス可能に
