from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import tensorflow as tf
# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define a flask app
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['PREDICTED_FOLDER'] = './static/predicted_images'
model=load_model('model.h5')
def age_prediction(img_path,model):
    img=cv2.imread(img_path)
    # img=cv2.resize(img,(512,256))
    h,w,c=img.shape
    t=h//48
    face_img=img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face=0
    c=0
    print(t)
    for (x, y, w, h) in faces:
        face=gray_img[y:y+h,x:x+w]
        face=cv2.resize(face,(48,48))
        face=face/255.0
        face=np.reshape(face,(1,48,48,1,))
        age=model.predict(face)[0][0]        
        print(age)
        face_img=cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        age_disp='{0} years'.format(int(np.round(age)))
        face_img=cv2.putText(face_img,age_disp,(x+5,y-10), font, 1, (0, 255, 0),2)
        c=c+1
    print(c)
#     s=cv2.resize(img,(48,48))
#     plt.imshow(face_img)
    return face_img


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    age=19
    img='no'
    f_name='no'
    if request.method == 'POST':
        # check if the post request has the file part
        for i in request.files:
            print(i)
        if 'image_file' not in request.files:
            flash('No file part')
            print('No selected file')
            return redirect(request.url)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        f = request.files['image_file']
        if f.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     return redirect(url_for('download_file', name=filename))
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # Get the file from post request
        # f = request.files['file']
        # request.
        # Save the file to ./uploads
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)

        # Make prediction
        img = age_prediction(path, model)    
        # print(age)
        pred_path=os.path.join(app.config['PREDICTED_FOLDER'],f.filename)
        cv2.imwrite(pred_path,img)

        f_name=pred_path
        # return result 
    return render_template('index.html',file_name=f_name)

if __name__ == '__main__':
    app.run(debug=True)