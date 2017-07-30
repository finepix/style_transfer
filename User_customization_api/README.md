## Guide for INetwork.py
###render()参数说明
> base_image_path                 内容图片的地址（本地地址，从文件中读取图片）

> style_reference_image_paths     风格图图片（同上，这里主要需要给出的参数为list，因为考虑到一次使用多个风格图）

> content_weight = 0.0025         内容图的渲染的权重（权重越大结果图越接近原图）

> style_weight=[1]                风格图的权重（同上）

> tv_weight = 8.5e-5              总体的损失

> style_scale=1                   风格损失比例

> style_masks = None              不需要渲染的区域为黑色，需要的区域为白色

> color_mask = None

> color='False'                   需要保留原图的颜色与否

> pool = 'max'                    卷积神经网络的池化层类型（对于原图差异明显，轮廓分明的推荐使用maxpooling）

> img_size=400                    图片的宽（会按比例对图片放缩用来计算，图片太大显存会爆掉）

> model_name = 'vgg16'            model的名字，分为vgg16与vgg19，是imagenet上使用的一系列卷积网络

> min_improvement = 5.0           每一次迭代会相比与上一次迭代有一次百分比提高，低于这个设定值，那么程序将会提前给出结果

> rescale_image = 'False'         不管，图片放缩的，默认可以

> num_iter = 5                    程序迭代次数，min_improvement共同决定程序的结束


###普通渲染
```
render('images/inputs/content/ancient_city.jpg',['images/inputs/style/blue_swirls.jpg'])
```
只需要内容图片地址以及风格图片的地址即可

###使用遮罩层（渲染的时候）
```
render('images/inputs/content/ancient_city.jpg',['images/inputs/style/blue_swirls.jpg'],style_mask=['path_to_style_mask'])
```
注意mask给出也是list，和前面的style_path统一

###保存原图的颜色（渲染的时候）
```
render('images/inputs/content/ancient_city.jpg',['images/inputs/style/blue_swirls.jpg'],color=True)
```
将color设置为true即可,也可以添加color_mask指定特定的区域保留颜色(mask为图片，黑白图片，且黑色部分表示保留颜色)
```
render('images/inputs/content/ancient_city.jpg',['images/inputs/style/blue_swirls.jpg'],color=True,color_mask='path_to_color_mask')
```

## Guild for mask_transfer.py (使用遮罩，渲染结束后生成的图)

### 使用说明
> 这个和上面的区别开，这个方法是用于一个图片已经渲染之后，我发现我没有使用遮罩现在我想用遮罩，那么不必要再渲染一次，那么调用这个方法

###函数申明
```
def mask(
        content_image,
        generated_image,
        content_mask)
```
</b>返回</b>：图片存放的位置,默认存放在'/tmp/smg/transfer/'下面，涉及到一个问题，就是文件的访问权限，写的权限一般是普通用户（hdu_admin）不具备的
，所以启动服务器的时候要么给sudo，要么chmod
###参数说明

> content_image: 内容图

> generated_image: 生成图

> content_mask: 遮罩

###调用说明

> 参数传递：将图片的地址传过来

## Guild for color_transfer.py (颜色保护，渲染结束后生成的图)

### 使用说明
> 和上面的mask_transfer一样的（还是注意文件写入权限），直接将图片保存到/tmp/smg/transfer/目录下面去

###函数申明
```python
def color_preserve(
        content_image,                                  # path to content image
        generated_image,                                # path to generated image
        hist_match=0                                    # Perform histogram matching for color matching 1 for histogram 0 for original
):
```
</b>返回</b>：处理之后的图片存放位置

###参数说明

> content_image: 内容图

> generated_image: 生成图

> hist_match: 是否使用柱状图的方式拟合，0表示直接使用原图的像素点替换，1表示使用柱状图拟合，得到一个相近的结果

###调用说明

> 参数传递：将图片的地址传过来


## Require
numpy
scipy
keras
tensorflow-gpu
pillow
