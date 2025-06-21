from flask import Flask, render_template, request, redirect, url_for,flash,session
from keras.models import load_model
import numpy as np
import cv2
import os
from flask import Flask
from werkzeug.utils import secure_filename
import os
from tensorflow import keras
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import dlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import mysql.connector
conn=mysql.connector.connect(host="localhost",user="root",password="mypc",autocommit=True)
mycursor=conn.cursor(dictionary=True,buffered=True)
mycursor.execute("create database if not exists deepfake")
mycursor.execute("use deepfake")
mycursor.execute("create table if not exists deep(id int primary key auto_increment,cname varchar(255),email varchar(30) unique,cpassword text)")

app = Flask(__name__)
app.secret_key = 'super secret key'
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def main():
  return render_template ("index.html")

@app.route('/registration',methods =['GET', 'POST'])
def registration():
  if request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'password' in request.form:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        mycursor.execute("SELECT * FROM deep WHERE email = '"+ email +"' ")
        account = mycursor.fetchone()
        if account:
            flash('You are already registered, please log in')
        else:
            
            mycursor.execute("insert into deep values(NULL,'"+ name +"','"+ email +"','"+ password +"')")
            msg=flash('You have successfully registered !')
            return render_template('registration.html',msg=msg)
  return render_template("registration.html")

@app.route('/login',methods =['GET', 'POST'])
def login():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        mycursor.execute("SELECT * FROM deep WHERE email = '"+ email +"' AND cpassword = '"+ password +"'")
        account = mycursor.fetchone()
        print(account)
        if account:
            session['loggedin'] = True
            session['email'] = account['email']
            msg = flash('Logged in successfully !')
                
            return redirect(url_for('home'))
        else:
            msg = flash('Incorrect username / password !')
            return render_template('login.html',msg=msg)
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/upload_video',methods =['GET', 'POST'])
def Video():
  return render_template ("Upload_Video.html")

@app.route('/upload_image',methods =['GET', 'POST'])
def image():
    return render_template('Upload_image.html')

model = tf.keras.models.load_model('model/deepfake-detection-tensor.h5')

def prediction (filepath):
    input_shape = (128, 128, 3)
    pr_data = []
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(filepath)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                data = data.reshape(-1, 128, 128, 3)
                print((model.predict(data) > 0.5).astype("int32"))
                outt=''
    if ((model.predict(data) > 0.5).astype("int32")[0][0] == 0 ):
        return 'Real'
    else:
        return'Fake'


@app.route('/predict_video',methods =['GET', 'POST'])
def upload_video():
	file = request.files['file']
	filename = secure_filename(file.filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	filepath = "static/"+filename
	preds = prediction(filepath)
	return render_template("Display_Video.html",prediction =preds ,video_path = filename)



@app.route('/Predict_image', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join (app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = prediction(file_path)
    return render_template("Display_image.html",prediction = preds, img_path= f.filename )

if __name__ =="__main__":
  app.run(debug=True)
