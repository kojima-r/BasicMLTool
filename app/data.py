# data.py
from flask import Blueprint
data1 = Blueprint("data1", __name__,
    static_url_path='/data1', static_folder='./data1'
)
data2 = Blueprint("data2", __name__,
    static_url_path='/data2', static_folder='./data2'
)
