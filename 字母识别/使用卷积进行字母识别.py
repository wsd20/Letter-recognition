import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 导入数据
data = pd.read_csv("./A_Z Handwritten Data.csv").astype('float32')
# 显示头五条数据
data.head()
print(data.shape)   # 显示数据形状是(372450, 785)，图片的大小为28*28的灰度图，第一列为标签

# 将第一列改名为标签
data.rename(columns={'0':'label'}, inplace=True)
data.head()


# 查看我们有多少标签
data.label.nunique()

#  设置数据集
y = data.label
x = data.drop('label',axis=1)

print(y.shape)
print(x.shape)


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

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建模型
model = tf.keras.models.Sequential([

    # 卷积核个数为32，卷积核尺寸为3*3，滑动步长为2，输入为28*28*1的单通道灰度图，池化层那里默认全零填充(输入特征图与输出特征图形状一致)
    # 池化层用于减少特征数据量，最大池化可提取图片纹理，均值池化可保留背景特征。
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
# checkpoint_save_path = "./checkpoint/alphabet.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)

#开始训练模型, 调用回调函数，当再次运行时，检查是否有上次训练保存的模型，如果有，在上次训练的模型的基础上继续训练
history = model.fit(x_train,y_train,batch_size=32,epochs=5,
                    validation_data=(x_test,y_test),validation_freq=1,
                    )
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


# 对预测进行图像展示

#使用图片展示预测结果
letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
                7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
                14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
                21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


preds = model.predict(x_test)

y_train = tf.keras.utils.to_categorical(y_train,num_classes=26,dtype=int)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=26,dtype=int)

X_test_ = x_test.reshape(x_test.shape[0], 28, 28, 1)

fig, axis = plt.subplots(3, 3, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test_[i].reshape(28,28),cmap='gray')
    ax.axis('off')
    ax.set(title = f"Real Alphabet : {letters_dict[y_test[i].argmax()]}\nPredicted Alphabet : {letters_dict[preds[i].argmax()]}")


plt.imshow(X_test_[1].reshape(28,28),cmap='gray')

plt.show()