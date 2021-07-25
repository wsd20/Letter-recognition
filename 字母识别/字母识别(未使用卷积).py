import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 导入数据
data = pd.read_csv("./大作业/A_Z Handwritten Data.csv").astype('float32')
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
x  = x/255.0

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(26,activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])




#保存模型
checkpoint_save_path = "./checkpoint/alphabet.ckpt"


if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)



#开始训练模型
history = model.fit(x,y,batch_size=32,epochs=5,
                    validation_data=(x_test,y_test),
                    validation_freq=1)
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
