# Common imports
import numpy as np
from PIL import Image

# TensorFlow imports
# may differs from version to versions
import tensorflow as tf
from tensorflow import keras

# OpenCV
import cv2

# Flask
from flask import request
from app import db
from app.model.user import User
##from sqlalchemy import text


# def get_class_name_from_db():
#     try:
#         sql = text('SELECT * FROM class_name')
#         result = db.engine.execute(sql)
#         class_names = []
#         for row in result:
#             class_names.append(row[1])
#         return class_names
#     except Exception as e:
#         print(e)
#         return None    

def predict():
    # opencv object that will detect faces for us
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load model to face classification
    model_name = 'face_classifier.h5'

    face_classifier = keras.models.load_model(f'../models/{model_name}')
    class_names = ['abizar', 'bintang', 'muchdor']
    # class_names = get_class_name_from_db()

    def get_extended_image(img, x, y, w, h, k=0.1):
        '''
        Function, that return cropped image from 'img'
        If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
        If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
        After getting the desired image resize it to 250x250.
        And converts to tensor with shape (1, 250, 250, 3)

        Parameters:
            img (array-like, 2D): The original image
            x (int): x coordinate of the upper-left corner
            y (int): y coordinate of the upper-left corner
            w (int): Width of the desired image
            h (int): Height of the desired image
            k (float): The coefficient of expansion of the image

        Returns:
            image (tensor with shape (1, 250, 250, 3))
        '''

        # The next code block checks that coordinates will be non-negative
        # (in case if desired image is located in top left corner)
        if x - k*w > 0:
            start_x = int(x - k*w)
        else:
            start_x = x
        if y - k*h > 0:
            start_y = int(y - k*h)
        else:
            start_y = y

        end_x = int(x + (1 + k)*w)
        end_y = int(y + (1 + k)*h)

        face_image = img[start_y:end_y,
                        start_x:end_x]
        face_image = tf.image.resize(face_image, [250, 250])
        # shape from (250, 250, 3) to (1, 250, 250, 3)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    image = Image.open(request.files['image'])
    npimg = np.array(image)
    gray = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # for each face on the image detected by OpenCV
        # get extended image of this face
        face_image = get_extended_image(npimg, x, y, w, h, 0.5)

        # classify face and draw a rectangle around the face
        # green for positive class and red for negative
        result = face_classifier.predict(face_image)
        prediction = class_names[np.array(
            result[0]).argmax(axis=0)]  # predicted class
        confidence = np.array(result[0]).max(axis=0)  # degree of confidence

    # display the resulting frame
    return prediction


## UNDER DEVELOPMENT
# def register_face():
#     return 'registerFace'

# def train():


#     train_image_folder = os.path.join('datasets', 'face_dataset_train_aug_images')
#     test_image_folder = os.path.join('datasets', 'face_dataset_test_images')
#     img_height, img_width = 250, 250  # size of images
#     num_classes = 3 

#     # Training settings
#     validation_ratio = 0.15  # 15% for the validation
#     batch_size = 16

#     AUTOTUNE = tf.data.AUTOTUNE


#     # Train and validation sets
#     train_ds = keras.preprocessing.image_dataset_from_directory(
#         train_image_folder,
#         validation_split=validation_ratio,
#         subset="training",
#         seed=42,
#         image_size=(img_height, img_width),
#         label_mode='categorical',
#         batch_size=batch_size,
#         shuffle=True)

#     val_ds = keras.preprocessing.image_dataset_from_directory(
#         train_image_folder,
#         validation_split=validation_ratio,
#         subset="validation",
#         seed=42,
#         image_size=(img_height, img_width),
#         batch_size=batch_size,
#         label_mode='categorical',
#         shuffle=True)


#     # Test set
#     test_ds = keras.preprocessing.image_dataset_from_directory(
#         test_image_folder,
#         image_size=(img_height, img_width),
#         label_mode='categorical',
#         shuffle=False)


#     class_names = test_ds.class_names
#     class_names


#     """
#     ## Modelling with ResNet50
#     """


#     base_model = keras.applications.ResNet50(weights='imagenet',
#                                             include_top=False,  # without dense part of the network
#                                             input_shape=(img_height, img_width, 3))

#     # Set layers to non-trainable
#     for layer in base_model.layers:
#         layer.trainable = False

#     # Add custom layers on top of ResNet
#     global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
#     output = keras.layers.Dense(num_classes, activation='sigmoid')(global_avg_pooling)

#     face_classifier = keras.models.Model(inputs=base_model.input,
#                                         outputs=output,
#                                         name='ResNet50')
#     face_classifier.summary()


#     # ModelCheckpoint to save model in case of interrupting the learning process
#     checkpoint = ModelCheckpoint("models/face_classifier.h5",
#                                 monitor="val_loss",
#                                 mode="min",
#                                 save_best_only=True,
#                                 verbose=1)

#     # EarlyStopping to find best model with a large number of epochs
#     earlystop = EarlyStopping(monitor='val_loss',
#                             restore_best_weights=True,
#                             patience=3,  # number of epochs with no improvement after which training will be stopped
#                             verbose=1)

#     callbacks = [earlystop, checkpoint]


#     face_classifier.compile(loss='categorical_crossentropy',
#                             optimizer=keras.optimizers.Adam(learning_rate=0.01),
#                             metrics=['accuracy'])


#     """
#     ## Training
#     """


#     epochs = 5

#     history = face_classifier.fit(
#         train_ds,
#         epochs=epochs,
#         callbacks=callbacks,
#         validation_data=val_ds)

#     face_classifier.save("models/face_classifier.h5")



#     return 'train'
