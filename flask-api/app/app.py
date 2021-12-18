# # Common imports
# import numpy as np
# from PIL import Image

# # TensorFlow imports
# # may differs from version to versions
# import tensorflow as tf
# from tensorflow import keras

# # OpenCV
# import cv2

# # Flask
# from flask import Flask, request
# from flask_mysqldb import MySQL
# app = Flask(__name__)

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'face_schema'



# @app.route('/', methods=['GET'])
# def hello_world():
#     return 'Hello World!'

# @app.route('/predict', methods=['POST'])
# def predict():

#     # opencv object that will detect faces for us
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Load model to face classification
#     model_name = 'face_classifier.h5'

#     face_classifier = keras.models.load_model(f'../models/{model_name}')
#     class_names = ['abizar', 'bintang', 'muchdor']

#     def get_extended_image(img, x, y, w, h, k=0.1):
#         '''
#         Function, that return cropped image from 'img'
#         If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
#         If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
#         After getting the desired image resize it to 250x250.
#         And converts to tensor with shape (1, 250, 250, 3)

#         Parameters:
#             img (array-like, 2D): The original image
#             x (int): x coordinate of the upper-left corner
#             y (int): y coordinate of the upper-left corner
#             w (int): Width of the desired image
#             h (int): Height of the desired image
#             k (float): The coefficient of expansion of the image

#         Returns:
#             image (tensor with shape (1, 250, 250, 3))
#         '''

#         # The next code block checks that coordinates will be non-negative
#         # (in case if desired image is located in top left corner)
#         if x - k*w > 0:
#             start_x = int(x - k*w)
#         else:
#             start_x = x
#         if y - k*h > 0:
#             start_y = int(y - k*h)
#         else:
#             start_y = y

#         end_x = int(x + (1 + k)*w)
#         end_y = int(y + (1 + k)*h)

#         face_image = img[start_y:end_y,
#                         start_x:end_x]
#         face_image = tf.image.resize(face_image, [250, 250])
#         # shape from (250, 250, 3) to (1, 250, 250, 3)
#         face_image = np.expand_dims(face_image, axis=0)
#         return face_image

#     image = Image.open(request.files['image'])
#     npimg = np.array(image)
#     gray = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=5,
#         minSize=(100, 100),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )

#     for (x, y, w, h) in faces:
#         # for each face on the image detected by OpenCV
#         # get extended image of this face
#         face_image = get_extended_image(npimg, x, y, w, h, 0.5)

#         # classify face and draw a rectangle around the face
#         # green for positive class and red for negative
#         result = face_classifier.predict(face_image)
#         prediction = class_names[np.array(
#             result[0]).argmax(axis=0)]  # predicted class
#         confidence = np.array(result[0]).max(axis=0)  # degree of confidence

#     # display the resulting frame
#     return prediction


# if __name__ == '_main_':
#     app.run(port=3000, debug=True) 

