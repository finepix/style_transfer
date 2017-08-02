#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:f-zx 
@file: transfer_models.py 
@time: 2017/08/02 
"""
import os

class tf_models:
    # 初始化
    def __init__(self):
        self.models_dir='./Common_transfer_api/models'
        self.models = {}
        self.generate_models()

    # 加载所有的模型
    def generate_models(self):
        for f in os.listdir(self.models_dir):
            file_name = os.path.basename(f)
            if file_name.endswith('.model'):
                self.models[file_name.split('.')[0]] = os.path.abspath(f)

        print 'load models:'
        print self.models

    # 获取模型
    def get_models(self,model_name):
        return self.models[model_name]


if __name__ == '__main__':
    t = tf_models()
    print t.get_models('mosaic')