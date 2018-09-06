# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image #安装PIL : pip install Pillow
import os
import matplotlib.pyplot as plt
import tensorflow as tf

'''
    1、将图片变为数组
    2、python 内置方法打开图片
'''

filePath = "E:\\picture"  #要读取的图片路径
FILENAMES = []  #存放文件夹名
LABELS = []     #存放文件对应的标签

singe_floder_name='1data' #单个文件夹多张图片的保存的文件名称
store_file_path="E:\\"    #保存路径
many_floder_name='4data'  #多个文件夹多张图片的保存的文件名称

#打开图片并变成数组后在重新reshape
def openImageToArrayWithReshap(category,imagename,shape,filepath=filePath):
    image = Image.open(os.path.join(filepath,category,imagename))
    image_arr = np.array(image,dtype=np.float64)
    # print("原始的image_arr=",image_arr)
    image_arr = np.reshape(image_arr,newshape=shape)
    # print("图片数组shape:",image_arr.shape)

    return image_arr

#把打开的图片变成数组
def openImageToArray(category,imagename,filepath=filePath):
    image = Image.open(os.path.join(filepath,category,imagename))
    image_arr = np.array(image)
    # print("图片数组shape:", image_arr.shape)
    # print("图片数组type:", type(image_arr))
    return image_arr

a = openImageToArray("one","11.png")
b = openImageToArrayWithReshap("two","22.png",[1,6,6,3])

#显示图片
def showpic(image_arr):
    #print(image_arr)
    plt.imshow(image_arr)
    plt.show() #原理：plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的函数显示出来。
showpic(a)


##########################################################
## 批量取出文件夹中的文件名 和类别标签，并存入一个列表中
##########################################################

#根据类别获取标签
def getLabel(category):
    label_dict = {"one":1,"two":2,"three":3,"four":4}
    return label_dict[category]

#获取图像的名称
def getFileName(category,file_dir=filePath):
    filenames = []  #存放文件名
    labels = []  #存放文件对应的标签
    '''
    os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
        top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
            root 所指的是当前正在遍历的这个文件夹的本身的地址
            dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
            files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)

            topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。
                     如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
            onerror -- 可选， 需要一个 callable 对象，当 walk 需要异常时，会调用。
            followlinks -- 可选， 如果为 True，则会遍历目录下的快捷方式(linux 下是 symbolic link)实际所指的目录(默认关闭)。
    '''
    print("category=",category)
    for root,dirs,files in os.walk(os.path.join(file_dir,category)):
        print("root=",root)
        print("dirs=",dirs)
        print("files=",files)
        for file in files:
            filenames.append(file)
            labels.append(getLabel(category))
    return filenames,labels

#定义子目录类别
dir_category=["one","two","three","four"]

#根据子目录，获取里面的文件
for category in dir_category:
    filenames,labels = getFileName(category,filePath)
    FILENAMES.append(filenames)
    LABELS.append(labels)



#单文件夹多图片  category为类别 filenames为list
def image_arrays(category,filenames,shape,file_path=filePath):
    imageArrays = []
    for filename in filenames:
        imamgeArray = openImageToArrayWithReshap(category,filename,shape,file_path)
        imageArrays.append(imamgeArray)
    #将图片数组连接起来
    arrays = np.row_stack(imageArrays)
    return arrays


#多文件夹多图片  category为类别 filenames为list,这个也适合单文件夹多文件
def folder_image_arrays(FILENAMES,shape,file_path=filePath):
    imageArrays = []
    labelsArrays = []
    for i in range(len(FILENAMES)):
        category = dir_category[i]
        for filename in FILENAMES[i]:
            index = FILENAMES[i].index(filename)
            imamgeArray = openImageToArrayWithReshap(category,filename,shape,file_path)
            imageArrays.append(imamgeArray)
            print("imageArrays=",imageArrays)
            labelsArrays.append(LABELS[i][index])
     #将图片数组连接起来
    arrays = np.row_stack(imageArrays)
    #将图片的标签连接起来
    lables = np.row_stack(labelsArrays)
    print("按行堆叠后的数组=",arrays)
    print("按行堆叠后的标签数组=",lables)
    return arrays,lables

'''
写文件的方法
'''
#转换成int64位的特征
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成浮点型的属性
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#若想保存为数组，则要改成value=value即可

#转换成byte特征
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 写文件
def writeTF(file_path,images, labels, name):
    #通过检查label的数量是否与特征数组的第一维数是否一样，判读传入数据是否有误
    num_examples = len(labels)
    if len(labels) != num_examples:
        raise ValueError("图片数量 %d 不匹配标签数量 %d." %
                         (len(labels), num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(file_path, name + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw  = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))

        #序列化输出到磁盘对应文件中
        writer.write(example.SerializeToString())

# 转为灰度图片，shape传入形状的元组
def grey(imageArrays,shape):
    imageArrays = np.reshape(np.average(imageArrays,axis=3),shape)
    return imageArrays


#####################################################
## 把多个文件夹下的多张图片合并起来并且写成TFRecord
#####################################################
images,labels = folder_image_arrays(FILENAMES,[1,6,6,3],filePath)
writeTF(store_file_path,images,labels,many_floder_name)









