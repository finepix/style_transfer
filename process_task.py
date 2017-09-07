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
import thread
from Common_transfer_api.render import render_with_model_file


from transfer_models import tf_models

# 服务器地址
base_url = 'http://101.132.77.137:8080/'


def pop():
    '''
{
    "code": 0,
    "msg": "",
    "data": {
        "id": "1504765355806",
        "type": "model",
        "originalPic": "src/pic.jpg",
        "model_name": "mosaic"
    }
}
    :return: 
    '''
    pop_url = base_url + 'tasks/pop'
    data = requests.get(pop_url).text
    obj = json.loads(data)

    # ctn = True if len(obj['code']) == 0 else False
    return obj

##文件保存文件夹
tmp_dir = './tmp/'

def download_file(url,path=tmp_dir):
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



def commit_task():
    pass

def model_task(data):
    '''
{
    "code": 0,
    "msg": "",
    "data": {
        "id": "1504765355806",
        "type": "model",
        "originalPic": "src/pic.jpg",
        "model_name": "mosaic"
    }
}    
    
    :param task: 
    :return: 
    '''
    # 下载文件
    file_path = download_file(data['originalPic'])
    # 渲染处理
    model_file = tf_models.get_models(data['model_name'])
    res_file = render_with_model_file(file_path, model_file=model_file)
    return res_file

if __name__ == '__main__':
    tf_models = tf_models()
    # 获取锁
    lock = thread.allocate_lock()
    while True:
        # 有任务
        task = pop()
        if task['code'] == 0 :
            lock.acquire()
            print 'get to a task:'
            if task['data']['type'] == 'model':
                # 上传文件以及回调函数设置
                res_file = model_task(task['data'])
                ## 改名字
                key = os.path.basename(res_file)
                commit_task()
            lock.release()
        else:
            print 'no tasks,wait for 0.5s'
            time.sleep(0.5)
