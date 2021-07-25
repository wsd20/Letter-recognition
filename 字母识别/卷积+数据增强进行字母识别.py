import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 导入数据
data = pd.read_csv("./大作业/A_Z Handwritten Data.csv").astype('float32')
# 显示头五条数据
data.head()
print(data.shape)   # 显示数据形状是(372450, 785)，图片的大小为28*28的灰度图，第一列为标签

# 将第一列改名为标签
data.rename(columns={'0':'label'}, inplace=True)
data.head()


#  设置数据集
y = data.label
x = data.drop('label',axis=1)



#分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
x_train, x_test = x_train / 255.0, x_test / 255.0



# 将数据从一维数组转换为二维，方便用图片展示
x_train= x_train.values.reshape(297960,28,28)
x_test= x_test.values.reshape(74490,28,28)

x_train.shape,x_test.shape


# 将数据转换为二维数组,展示图像
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
plt.subplot(1, 2, 1)
plt.imshow(x_train[0],cmap='gray')
plt.title('示例图片1')
plt.subplot(1, 2, 2)
plt.imshow(x_train[1],cmap='gray')
plt.title('示例图片2')
plt.show()

# 给数据增加一个维度，使数据和网络结构匹配,增加一个单通道，这个单通道是灰度值
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train.shape,x_test.shape


# 数据增强,提高模型的泛化能力
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)



# 构建模型
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(26,activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


# 断点续训,保存模型
checkpoint_save_path = "./checkpoint/alphabet.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)


#开始训练模型, 调用回调函数，当再次运行时，检查是否有上次训练保存的模型，如果有，在上次训练的模型的基础上继续训练
'''
flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')：
接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
'''
history = model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),epochs=5,
                    validation_data=(x_test,y_test),validation_freq=1,
                    callbacks=[cp_callback])
model.summary()



# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
