import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

def img_to_array(imgormask,name,size):
    img=Image.open(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train',imgormask,'img',name)).convert('L')
    img = img.resize(size)
    img_array = np.array(img)

    
    #img_array = np.expand_dims(img_array[:,:1], axis=0)
    img_array = img_array /255.
    binary_array = np.round(img_array)
    
    #np.savetxt('output.txt',binary_array)
    return binary_array

def generators(targetsize,batchsize,
               train_image_dir,train_mask_dir
               ,val_image_dir,val_mask_dir
               ,test_image_dir,test_mask_dir
               ):
    class_mode = None
    colormode = 'grayscale'

    seed=42
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255
                                                                ,horizontal_flip=True
                                                                ,zoom_range=0.3
                                                                ,rotation_range=40)
    
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255
                                                              ,horizontal_flip=True
                                                              ,zoom_range=0.3
                                                              ,rotation_range=40)
    
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_image_generator = train_gen.flow_from_directory(directory=train_image_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize
                                                          )
    train_mask_generator = train_gen.flow_from_directory(directory=train_mask_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize
                                                          )
    train_generator = zip(train_image_generator,train_mask_generator)


    val_image_generator = val_gen.flow_from_directory(directory=val_image_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize)
    val_mask_generator = val_gen.flow_from_directory(directory=val_mask_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=batchsize)
    validation_generator = zip(val_image_generator,val_mask_generator)


    test_image_generator = test_gen.flow_from_directory(directory=test_image_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=1)
    test_mask_generator = test_gen.flow_from_directory(directory=test_mask_dir,
                                                          seed=seed,
                                                          class_mode=class_mode,
                                                          color_mode=colormode,
                                                          target_size=targetsize,
                                                          batch_size=1)
    test_generator = zip(test_image_generator,test_mask_generator)


    return train_generator,validation_generator,test_generator

# def load_images(imgormask,dir,size):
#     images = []
#     for filename in os.listdir(dir):
#         img_path = os.path.join(dir,filename)
#         img_array = img_to_array(imgormask,filename,size)
#         # if img_array.shape==(128,128):
#         #     img_array = np.expand_dims(img_array,axis=-1)
#         #     img_array = np.tile(img_array,(1,1,3))
#         images.append(img_array)
#     images = np.array(images)
#     return images





def main():
    pass


if __name__ == "__main__":
    main()


