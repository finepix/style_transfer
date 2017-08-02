#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:f-zx 
@file: process_task.py 
@time: 2017/08/02 
"""

import requests
import json
import os
import time
from Common_transfer_api.render import render_with_model_file

from qiniu import Auth, put_file, etag
import qiniu.config
from qn_setting import Qn_setting
from transfer_models import tf_models

# 服务器地址
base_url = 'http://60.176.42.123:5000/'

# 上传文件到七牛后， 七牛将文件名和文件大小回调给业务服务器。
policy = {
    'callbackUrl': 'http://60.176.42.123:5000/tasks/callback',
    'callbackBody': 'filename=$(fname)'
}

def test_tasks(url):

    # 测试tasks请求
    data = requests.get(url).text
    print data

def query():
    '''
    {
  "done": [],
  "tasks": [
    {
      "model": "mosaic",
      "type": "model",
      "url": "https://oi3qt7c8d.qnssl.com/res.jpg"
    },
    {
      "model": "mosaic",
      "type": "model",
      "url": "https://oi3qt7c8d.qnssl.com/res.jpg"
    }
  ]
}
    :return:   true 表示有任务
                false 表示没有任务，那么继续查询
    '''
    query_url = base_url + 'tasks/query'
    data = requests.get(query_url).text
    print data
    obj = json.loads(data)
    return  True if len(obj['tasks']) == 0 else False

def pop():
    pop_url = base_url + 'tasks/pop'
    data = requests.get(pop_url).text
    obj = json.loads(data)
    return obj

##文件保存文件夹
tmp_dir = './tmp/'

def download_qb(url,path=tmp_dir):
    '''
            下载文件，默认保存到tmp文件夹之中
    :param url: 文件存储地址
    :return: 文件存放地址
    '''
    if os.path.exists(path) is False:
        os.mkdir(path)
    t = time.time()
    file_name = str(int(round(t * 1000))) + '.jpg'
    file_path = path + file_name

    file_data = requests.get(url,stream=True)
    with open(file_path,'wb') as f:
        for chunk in file_data.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    return file_path


def upload_qn(key,localfile):
    '''
    upload file to qn yun and set callback url and param
    :param key: 上传之后的key值默认和文件名一致
    :param localfile: 本地文件
    :return: 
    '''
    setting = Qn_setting()
    q = Auth(setting.access_key, setting.secret_key)

    token = q.upload_token(setting.bucket_name, key, 3600, policy)
    ret, info = put_file(token, key, localfile)
    print(info)


if __name__ == '__main__':
    tf_models = tf_models()
    # 有任务
    if not query():
        print 'get to a job:'
        job = pop()
        if job['type'] == 'model':
            # 下载文件
            file_path = download_qb(job['url'])
            # 渲染处理
            model_file = tf_models.get_models(job['model'])
            res_file  = render_with_model_file(file_path,model_file=model_file)
            # 上传文件以及回调函数设置
            ## 改名字
            key = os.path.basename(res_file)
            upload_qn(key,res_file)

