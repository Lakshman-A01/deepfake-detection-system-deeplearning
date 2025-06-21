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


ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'flv'}


def extract_frames(video_path, output_folder,  num_frames=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        # List all files in the folder
        files = os.listdir('static/facefolder')

        # Iterate through files and delete each one
        for file_name in files:
            file_path = os.path.join('static/facefolder', file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("Deletion complete.")

    except Exception as e:
        print(f"An error occurred: {e}")

    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if num_frames == 0:
        print("Error: num_frames must be greater than 0.")
        return

    frames_to_extract = range(0, frame_count, max(1, frame_count // num_frames))

    for frame_number in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(output_path, frame)

    cap.release()


def crop_faces(input_folder, output_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        # List all files in the folder
        files = os.listdir('static/frame')

        # Iterate through files and delete each one
        for file_name in files:
            file_path = os.path.join('static/frame', file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("Deletion complete.")

    except Exception as e:
        print(f"An error occurred: {e}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = image[y:y + h, x:x + w]
            output_path = os.path.join(output_folder, f"face_{filename.split('.')[0]}_{i}.jpg")
            cv2.imwrite(output_path, face_roi)

            

frames_output_folder = 'static/frame'
faces_output_folder = 'static/facefolder'


@app.route('/view')
def index():
    face_folder = 'static/frame'  # Change this to the path of your face images folder
    frame_folder = 'static/facefolder'  # Change this to the path of your frame images folder

    face_images = [img for img in os.listdir(face_folder) if img.endswith(".jpg")]
    frame_images = [img for img in os.listdir(frame_folder) if img.endswith(".jpg")]

    return render_template('bala.html', face_images=face_images, frame_images=frame_images)




model = tf.keras.models.load_model('model/deepfake-detection-tensor.h5')



def prediction(filepath):
    input_shape = (128, 128, 3)
    pr_data = []
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(filepath)
    frameRate = cap.get(5)
    
    data = None  # Initialize data outside the loop
    
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
                outt = ''
    extract_frames(filepath,faces_output_folder)
    crop_faces(faces_output_folder, frames_output_folder)
    face_folder = 'static/frame'  # Change this to the path of your face images folder
    frame_folder = 'static/facefolder'  # Change this to the path of your frame images folder

    face_images = [img for img in os.listdir(face_folder) if img.endswith(".jpg")]
    frame_images = [img for img in os.listdir(frame_folder) if img.endswith(".jpg")]
    
    # Check if data is not None before using it
    if data is not None:
        if ((model.predict(data) > 0.5).astype("int32")[0][0] == 0):
            return 'Real', face_images, frame_images
        else:
            return 'Fake', face_images, frame_images
  
@app.route('/predict_video',methods =['GET', 'POST'])
def upload_video():
	file = request.files['file']
	filename = secure_filename(file.filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	filepath = "static/"+filename
	preds, face, frame = prediction(filepath)
    
	return render_template("bala.html",prediction =preds, face_images= face, frame_images=frame)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')




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
            # msg=flash('You have successfully registered !')
            return redirect(url_for('login'))
  return render_template("reg.html")

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
                
            return redirect(url_for('video'))
        else:
            msg = flash('Incorrect username / password !')
            return render_template('login.html',msg=msg)
    return render_template('login.html')

if __name__ =="__main__":
  app.run(debug=True)