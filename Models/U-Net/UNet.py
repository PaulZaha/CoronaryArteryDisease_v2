import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2

def double_conv_block(x,n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x


def downsample_block(x,n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f,p


def upsample_block(x, conv_features, n_filters):
   x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   x = tf.keras.layers.concatenate([x, conv_features])
   x = tf.keras.layers.Dropout(0.3)(x)
   x = double_conv_block(x, n_filters)
   return x


def build_model():
    inputs = tf.keras.layers.Input(shape=(128,128,3))
    
    #Encoder Structure
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    #Bottleneck feature map
    bottleneck = double_conv_block(p4, 1024)
    
    #Decoder Structure
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    
    # outputs
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


   
def model_compiler(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  #"sparse_categorical_crossentropy",
                  metrics="accuracy")



def load_images_as_arrays(input_dir, target_size=(128, 128)):
    images = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Größenanpassung auf 128x128
                img = cv2.resize(img, target_size)
                images.append(img)

    img_array = np.array(images)
    return np.array(images)

def img_to_array(name):
    img=Image.open(os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','images',name))
    img = img.resize((128,128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array[:,:,:3], axis=0)
    img_array = img_array /255.
    return img_array

def predictor(name,model):
    predict_array = img_to_array(name)
    predictions = model.predict(predict_array)
    predictions = np.squeeze(predictions, axis=0) 
    predictions = np.argmax(predictions, axis=-1) 
    predictions = Image.fromarray(np.uint8(predictions*255)) 
    predictions = predictions.resize((128,128))
    predictions.save(name[:-4] + "_pred.jpg")

def main():
    unet = build_model()
    
    model_compiler(unet)

    image_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','images')
    mask_dir = os.path.join(os.getcwd(),'Dataset','arcade','stenosis','train','masks')
    x_train = load_images_as_arrays(image_dir)
    y_train = load_images_as_arrays(mask_dir)


    
    
    hist = unet.fit(x_train,y_train,
                    epochs=1,
                    batch_size=16,
                    verbose=1
                    )
    #tf.keras.saving.save_model(unet)
    predictor('1.png',unet)

if __name__ == "__main__":
    main()