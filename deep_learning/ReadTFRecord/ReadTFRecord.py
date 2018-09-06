# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
#定义存放到tfrecord中的数据字典格式
feat={
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64)}

#调用tf.TFRecordReader类的read函数，传人文件处理队列返回序列化对象
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    #解析序列化对象,解析TensorFlow中的单个Example原型,
    features = tf.parse_single_example(serialized_example,  features=feat)
    #解码序列化对象
    image = tf.decode_raw(features['image_raw'], tf.float32)

    #可选把我们原来的tf.int64数据类型转化为tf.int32
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    return image, label ,height,width,depth

################################
## 读取图像数据
################################
def get_all_records(FILE):
    with tf.Session() as sess:
        #返回一个用于文件处理的队列,tf.train.string_input_producer 可以传入多个文件名列表
        filename_queue = tf.train.string_input_producer([ FILE ])
        init_op = (tf.global_variables_initializer())

        #调用前面的read_and_decode函数
        image, label, height, width, depth   = read_and_decode(filename_queue)

        print("image=",image)
        print("label=",label)
        print("height=",height)
        print("width=",width)
        print("depth=",depth)
        print("image的形状：",image.shape)
        #生成一个shape=(1, 6, 6, 1)的image tensor
        image = tf.reshape(image, tf.stack([2,6,6,3]))
        print("image堆叠后的形状：",image.shape)
        sess.run(init_op)

        #tf.train.Coordinator生成一个读取文件线程的协调进程对象
        coord = tf.train.Coordinator()

        #开启读取文件线程 # 启动计算图中所有的队列线程
        threads = tf.train.start_queue_runners(coord=coord)

        featuredata=np.reshape(np.zeros(216),(2,6,6,3))
        print("featuredata=",featuredata)

        labeldata=np.reshape(np.zeros(1),(1))
        print("labeldata=",labeldata)

        #在当前的session下读取4次，每次读取一个图片
        for i in range(16):
            example, l = sess.run([image,label])
            print("example=",example.shape)
            print("L=",l.shape)
            featuredata=np.append(featuredata,example,axis=0)
            l=np.reshape(np.array(l),1)
            labeldata=np.append(labeldata,l,axis=0)

        #请求线程终止
        coord.request_stop()

        #等待线程完全终止
        coord.join(threads)

        #返回特征
        return featuredata[1:][:][:][:] , labeldata[1:]


if __name__ =="__main__":
    data , label=get_all_records('E:\\4data.tfrecords')
    print("data=",data.shape)
    print("label=",label)

