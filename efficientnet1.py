from tensorflow.keras.layers import Dense, Dropout, Layer, Embedding, \
    Input, LayerNormalization, MultiHeadAttention, Add, Flatten, Lambda, GlobalAveragePooling2D, \
    Reshape, Permute, multiply, Activation, GlobalMaxPooling2D, Concatenate, Conv2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import Sequential, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import efficientnet.tfkeras as efc
import numpy as np
import os
import struct

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
image_size = 300  # We'll resize input images to this size
patch_size = (2, 3)  # Size of the patches to be extract from the input images
num_patches = (32, 10)
projection_dim = 128
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 10
mlp_head_units = [2048, 512]  # Size of the dense layers of the final classifier

data_augmentation = Sequential(
    [
        preprocessing.Normalization(),
        preprocessing.Resizing(image_size, image_size),
        preprocessing.RandomFlip('horizontal'),
        preprocessing.RandomRotation(factor=0.2)
    ],
    name='data_augmentation',
)



# from tensorflow.keras.datasets.cifar100 import load_data
# (x_train, y_train), (x_test, y_test) = load_data()
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(4, 4))
# image = x_train[np.random.choice(range(x_train.shape[0]))]
# plt.imshow(image.astype("uint8"))
# plt.axis("off")
# plt.show()
#
# resized_image = tf.image.resize(
#     tf.convert_to_tensor([image]), size=(image_size, image_size)
# )
# patches = Patches(patch_size)(resized_image)
# print(f"Image size: {image_size} X {image_size}")
# print(f"Patch size: {patch_size} X {patch_size}")
# print(f"Patches per image: {patches.shape[1]}")
# print(f"Elements per patch: {patches.shape[-1]}")
#
# n = int(np.sqrt(patches.shape[1]))
# plt.figure(figsize=(4, 4))
# for i, patch in enumerate(patches[0]):
#     ax = plt.subplot(n, n, i + 1)
#     patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
#     plt.imshow(patch_img.numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()



def vit_model(patch_size=patch_size,
              num_patches=num_patches,
              projection_dim=projection_dim,
              transformer_layers=transformer_layers,
              num_heads=num_heads,
              mlp_head_units=mlp_head_units,
              transformer_units=transformer_units,
              input_shape=(224, 224, 3),
              num_classes=5):
    # inputs = Input(shape=input_shape)
    model = efc.EfficientNetB3(
        input_shape=input_shape,
        # classes=3,
        weights='imagenet',
        include_top=False,
        classes=1000
    )

    return model


if __name__ == "__main__":
    model = vit_model()
    model.summary()
    model.compile()
    model.save_weights("/home/pengyuzhou/workspace/tensorrtx_new/tensorrtx/psenet/models")
    # h5_to_pb("123.h5")
    # model.load_weights("123.h5")



    weights = model.get_weights()
    layers = model.weights
    print(weights)
    f = open(r"efficientnetb3_imagenet.wts", "w")
    f.write("{}\n".format(len(weights)))
    for weight,layer in zip(weights,layers):
        key = layer.name
        # print(key, weight.shape)
        if len(weight.shape) == 4:
            weight = np.transpose(weight, (3, 2, 0, 1))
            print(key)
            print(weight.shape)
        weight = np.reshape(weight, -1)
        f.write("{} {} ".format(key, len(weight)))
        for w in weight:
            f.write(" ")
            f.write(struct.pack(">f", float(w)).hex())
        f.write("\n")
