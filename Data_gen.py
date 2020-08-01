import os
import random
import numpy as np
import tensorflow as tf
from Utils import read_png
import json

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            pairs,
            size=(1024, 1024),
            n_classes=5,
            shuffle=True,
            seed=0,
            crop_size=None,
            augment=False,
            brightness_range=0.05,
            contrast_range=0.2,
            saturation_range=0.4,
            hue_range=0.05,
    ):
        self.size = size
        self.n_classes = n_classes
        self.crop_size = crop_size
        self.augment = augment
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.dataset = pairs  # pairs [i][post.json, post_img.png, pre.json, pre_img.png, mask]

        if shuffle:
            random.seed(seed)
            random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.augment:
            l_r = np.random.choice([True, False])
            u_d = np.random.choice([True, False])

        def augment(image, spatial_only):
            if l_r:
                image = tf.image.flip_left_right(image)
            if u_d:
                image = tf.image.flip_up_down(image)

            if not spatial_only:
                image = tf.image.random_brightness(image, self.brightness_range)
                image = tf.image.random_contrast(image, 1.0 - self.contrast_range, 1.0 + self.contrast_range)
                image = tf.image.random_saturation(image, 1.0 - self.saturation_range, 1.0 + self.saturation_range)
                image = tf.image.random_hue(image, self.hue_range)

            return image

        item = self.dataset[index % len(self.dataset)]
        pre = read_png(item[3])
        post = read_png(item[1])

        if self.augment:
            pre = augment(pre, False)
            post = augment(post, False)

        pre_post = tf.concat([pre, post], axis=-1)
        pre_post = tf.cast(pre_post, tf.float32) / 255.0
        pre_post = tf.image.resize(pre_post, self.size)
        pre_post = tf.expand_dims(pre_post, axis=0)

        mask = read_png(item[-1])

        if self.augment:
            mask = augment(mask, True)

        mask = tf.image.resize(mask, self.size, method="nearest")
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.cast(mask, tf.int32)
        mask = tf.minimum(mask, tf.constant(self.n_classes - 1, dtype=tf.int32))
        mask = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)

        if self.crop_size is not None:
            w = self.crop_size[0]
            h = self.crop_size[1]
            x = np.random.randint(0, pre_post.shape[1] - w)
            y = np.random.randint(0, pre_post.shape[2] - h)
            return pre_post[:, x:x + w, y:y + h, :], mask[:, x:x + w, y:y + h, :]
        else:
            return pre_post, mask

    def class_weights(self, beta=None):
        frequencies = np.zeros(self.n_classes, dtype=np.float32)
        for item in self.dataset:
            mask = read_png(os.path.join(self.mask_dir, item[2]))
            for i in range(self.n_classes):
                frequencies[i] += np.count_nonzero(mask == i)

        if beta is None:
            weights = frequencies ** -1
        else:
            frequencies /= len(self)
            weights = (1.0 - beta) / (1.0 - beta ** frequencies)
        weights /= np.mean(weights)
        return weights


class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs, directory, size=(1024, 1024)):
        self.size = size
        self.dataset = pairs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index % len(self.dataset)]

        pre = read_png(item[3])
        post = read_png(item[1])

        pre_post = tf.concat([pre, post], axis=-1)
        pre_post = tf.cast(pre_post, tf.float32) / 255.0
        pre_post = tf.image.resize(pre_post, self.size)
        pre_post = tf.expand_dims(pre_post, axis=0)
        mask = read_png(item[-1])
        mask = tf.image.resize(mask, self.size, method="nearest")
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.cast(mask, tf.int32)
        mask = tf.minimum(mask, tf.constant(self.n_classes - 1, dtype=tf.int32))
        mask = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)
        return pre_post, mask

def json2dict(filename):  # input: json url
    with open(filename) as f:
        j = json.load(f)
    return j

if __name__ == '__main__':
    os.chdir(r'/Users/czhui960/Documents/Segdataset')
    dict_pre_post_train1 = json2dict('./train/pairs_dict.json')
    dict_pre_post_train2 = json2dict('./train 2/pairs_dict.json')
    dir_list_train = []
    for key in dict_pre_post_train1.keys():
        dir_list_train+=dict_pre_post_train1[key]
    for key in dict_pre_post_train2.keys():
        dir_list_train+=dict_pre_post_train2[key]


    size = (1024,1024)
    crop_size = None
    train_gen = DataGenerator(
        dir_list_train,
        size=size,
        n_classes=5,
        shuffle=True,
        seed=1,
        crop_size=crop_size,
        augment=True)
    x,y = train_gen[0]
    print(x.shape,y.shape)
