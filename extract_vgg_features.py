import json
import tensorflow as tf
import os
import jieba
import numpy as np
from PIL import Image, ImageEnhance
import h5py
import matplotlib.pyplot as plt
import random

def main():
    IMG_SHAPE = (224, 224, 3)
    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE)
    model = tf.keras.Sequential(VGG16_MODEL.layers[:-5])
    model.trainable = False

    with open('./FM-CH-QA.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    train_json = json_data['train']
    val_json = json_data['val']

    # processing train_data
    train_h5_name = 'train_block5.h5'
    h5f = h5py.File(train_h5_name, 'w')
    key_id = set()
    for n, i in enumerate(train_json):

        if i['image_id'] not in key_id:
            key_id.add(i['image_id'])
            print(n, i['image_id'])

            img_path = getpath(i['image_id'])
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = np.array(img)
            img = (img / 255.0).astype('float32')
            img = img[tf.newaxis, ...]

            y = model(img).numpy().squeeze()
            h5f.create_dataset(i['image_id'], data=y)
    h5f.close()

    # processing val_data
    val_h5_name = 'val_block5.h5'
    h5f = h5py.File(val_h5_name, 'w')
    key_id = set()
    for n, i in enumerate(val_json):
        if i['image_id'] not in key_id:
            key_id.add(i['image_id'])
            print(n, i['image_id'])

            img_path = getpath(i['image_id'], mode='val')
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = np.array(img)
            img = (img / 255.0).astype('float32')
            img = img[tf.newaxis, ...]

            y = model(img).numpy().squeeze()
            h5f.create_dataset(i['image_id'], data=y)



    h5f.close()


def getpath(img_id, mode='train'):
    root_train = r'./Data/train2014/'
    root_val = r'./Data/val2014/'
    if mode == 'train':
        return os.path.join(root_train, 'COCO_train2014_'+(img_id.rjust(12, '0'))+'.jpg')
    else:
        return os.path.join(root_val, 'COCO_val2014_'+(img_id.rjust(12, '0'))+'.jpg')

if __name__ == '__main__':
    main()