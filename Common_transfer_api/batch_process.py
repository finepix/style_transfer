#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:f-zx 
@file: batch_process.py 
@time: 2017/07/25 
"""

import os
from . import render as rd

contents_path = 'img/content/'
models_path = 'models/'

contents = []
for c in os.listdir(contents_path):
    contents.append(os.getcwd()+ contents_path + c)

models = []
for m in os.listdir(models_path):
    if m.endswith('.model'):
        models.append(os.getcwd() + models + m)

for c in contents :
    for m in models:
        rd.render_with_model_file(c,m,output_file_path='output/')