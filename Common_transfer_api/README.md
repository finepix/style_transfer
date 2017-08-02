## 使用模型
###for render.py

    render_with_model_file('9.jpg','vgg_16','models/mosaic.model')


## 训练模型:
 下载[VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) 
```
cd <this repo>
mkdir pretrained
cp <your path to vgg_16.ckpt>  pretrained/
```

下载 [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip).
```
cd <this repo>
ln -s <your path to the folder "train2014"> train2014
```

Train the model of "wave":
```
python train.py -c conf/wave.yml
```

## Requirements 
- Python 2.7.x
- Now support Tensorflow = 1.2.0
