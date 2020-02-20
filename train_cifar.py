from model import resnet
import tensorflow as tf
import os
import numpy as np
import json

print('train on ', tf.config.experimental.list_physical_devices('GPU'))
# 使用GPU运算，并且设置内存增长为True
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

net = resnet.resnet_50()
net.build((None, 32, 32, 3))

net.summary()
net.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_dir = '/input0'

filenames = [os.path.join(train_dir,'data_batch_%d'%i) for i in range(1,6)]

train_data = {}

dict1 = unpickle(filenames[0])

train_data[b'data'] = dict1[b'data']
train_data[b'labels'] = dict1[b'labels']

for i in range(1, len(filenames)):
    file = filenames[i]
    dict = unpickle(file)
    train_data[b'data'] = np.concatenate((train_data[b'data'], dict[b'data']), axis=0)
    train_data[b'labels'] += dict[b'labels']

print(len(train_data[b'labels'])) #(50000)
print(train_data[b'data'].shape)  #(50000, 3072)

train_images = np.array(train_data[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1), dtype='float32')/255.0
train_images = tf.convert_to_tensor(train_images)
train_labels = tf.one_hot(train_data[b'labels'], 10)

val = unpickle(os.path.join(train_dir, 'test_batch'))
val_images = np.array(val[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1), dtype='float32')/255.0
val_images = tf.convert_to_tensor(val_images)
val_labels = tf.one_hot(val[b'labels'], 10)


import datetime
start = datetime.datetime.now()

history = net.fit(train_images, train_labels, batch_size=32, epochs=20, validation_data=(val_images, val_labels))
net.save_weights('ResNet_50.h5', save_format='h5')

print(history.history, file=open('history.txt', 'w'))

end = datetime.datetime.now()
print(end - start)