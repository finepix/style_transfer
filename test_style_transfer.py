#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:f-zx 
@file: test_style_transfer.py 
@time: 2017/08/02 
"""
import Common_transfer_api.render as rd

from Common_transfer_api.render import render_with_model_file

if __name__ == '__main__':

    rd.render_with_model_file(('Common_transfer_api/img/content/c2.jpg','Common_transfer_api/models/mosaic.model'))