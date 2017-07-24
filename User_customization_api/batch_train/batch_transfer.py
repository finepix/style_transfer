#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:f-zx 
@file: batch_transfer.py 
@time: 2017/07/23 
"""
import sys
sys.path.append('..')

import os
print(os.getcwd())

import INetwork as net


def main():
    absolute_path = os.getcwd()

    contents_path = 'content/'
    styles_path = 'styles/'
    c1_content_path = ''
    c1_mask_path = 'mask/m1.jpg'
    c11_content_path = ''
    c11_mask_path = 'mask/m11.jpg'

    contents = []
    styles = []

    contents_file_name = []
    styles_file_name = []
    mask = []
    for c in os.listdir(contents_path):
        contents.append(os.path.join(absolute_path , contents_path + c))
        contents_file_name.append(c.split('.')[0])
    for s in os.listdir(styles_path):
        styles.append(os.path.join(absolute_path , styles_path + s))
        styles_file_name.append(s.split('.')[0])
    

    for i in range(len(contents)):
        for j in range(len(styles)):
            prefix_file_name = contents_file_name[i] + '_' + styles_file_name[j]
            print 'render for \n %s \t %s' % (contents_file_name[i],styles_file_name[j])
            style = []
            style.append(styles[j])
            net.render(contents[i],
                   style,
                   out_file_prefix=prefix_file_name)

    for j in range(len(styles)):
        c1_content_path = contents.append(os.path.join(absolute_path, c1_content_path))
        print c1_content_path
        prefix  =  'm1_c1_' + styles_file_name[j]
        print 'render with mask for %s' % prefix
        style = []
        style_mask = []
        style_mask.append(c1_mask_path)
        style.append(styles[j])
        net.render(c1_content_path,
                    style,
                    style_masks=style_mask,
                    out_file_prefix=prefix)

    for j in range(len(styles)):
        c11_content_path = contents.append(os.path.join(absolute_path, c11_content_path))
        print c11_content_path
        prefix  =  'm1_c1_' + styles_file_name[j]
        print 'render with mask for %s' % prefix
        style = []
        style_mask = []
        style_mask.append(c11_mask_path)
        style.append(styles[j])
        net.render(c11_content_path,
                    style,
                    style_masks=style_mask,
                    out_file_prefix=prefix)

if __name__ == '__main__':
    main()


