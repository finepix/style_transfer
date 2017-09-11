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
from User_customization_api.color_transfer import color_preserve
from User_customization_api.INetwork import render


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
tmp_dir = '/root/ws/Shawn/tmp/'

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


def upload_file(res_file):
    upload_url = base_url + 'file/upload'
    files = {
        'render_file': open(res_file, "rb")
    }
    r = requests.post(upload_url, files=files)
    f = json.loads(r.text)['data']['render_file']

    print 'Render file is uploaded to url : %s' % f

    return f


def commit_task(task_id, res_file):
    commit_url = base_url + 'tasks/finish'
    f = upload_file(res_file)

    data = {
        'id':task_id,
        'result':f
    }
    r = requests.post(commit_url,data)
    print r.text


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

def color_task(data):
    '''
"data": {
    "id": "1504770912237",
    "type": "color",
    "originalPic": "http://101.132.77.137/resources/imgs/1504486077409.jpg",
    "renderPic": "http://101.132.77.137/resources/imgs/1504486077409.jpg"
}
    :param data: 
    :return: 
    '''
    source_pic_path = download_file(data['originalPic'])
    render_pic_path = download_file(data['renderPic'])
    res_file = color_preserve(source_pic_path,render_pic_path)
    return res_file



def customized_task(data):
    '''
"data": {
    "id": "1504771389744",
    "type": "customized",
    "originalPic": "http://101.132.77.137/resources/imgs/1504486077409.jpg",
    "stylePic": "http://101.132.77.137/resources/imgs/1504818106963.jpg"
}
    :param data: 
    :return: 
    '''
    pre_fix = "customized_"
    source_pic_path = download_file(data['originalPic'])
    style_pic_path = []
    style_pic_path.append(download_file(data['stylePic']))
    res_file = render(source_pic_path,style_pic_path,pre_fix)
    return res_file


if __name__ == '__main__':

    tf_models = tf_models()
    # 获取锁
    lock = thread.allocate_lock()
    while True:
        # 有任务
        res_file = ''
        task = pop()
        if task['code'] == 0:
            lock.acquire()
            print 'get to a task:'

            try:
                try:
                    if task['data']['type'] == 'model':
                        # 完成任务以及返回渲染后的图片地址
                        res_file = model_task(task['data'])

                    if task['data']['type'] == 'color':
                        res_file = color_task(task['data'])

                    if task['data']['type'] == 'customized':
                        res_file = customized_task(task['data'])

                    ## 获取任务id以及提交任务
                    key = task['data']['id']
                    commit_task(key, res_file)
                except:
                    print 'error when process task id: %s ,type: %s' % (task['data']['id'], task['data'['type']])
            except:
                print 'error ocurred when task proccessing.'


            lock.release()
        else:
            print 'no tasks,wait for 0.5s'
            time.sleep(0.5)
