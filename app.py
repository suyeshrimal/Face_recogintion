import os
import cv2
import shutil
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from flask import Flask,render_template,redirect,request,session,g,url_for
from datetime import datetime, date
import joblib
from dotenv import load_dotenv

load_dotenv()
# Initializing the flask app

app=Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Flask error handler
"""Code to be written"""

# Flask assign admin
@app.before_request
def before_request():
    g.user = None
    if 'admin' in session:
        g.user = session['admin']

# Current Date & Time
datetoday = date.today().strftime("%d-%m-%Y")
datetoday2 = date.today().strftime("%d %B %Y")

# Capture the video
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('UserList'):
    os.makedirs('UserList')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/{datetoday}.csv', 'w') as f:
        f.write('Name,ID,Section,Time')
if 'Registered.csv' not in os.listdir('UserList'):
    with open('UserList/Registered.csv', 'w') as f:
        f.write('Name,ID,Section')
if 'Unregistered.csv' not in os.listdir('UserList'):
    with open('UserList/Unregistered.csv', 'w') as f:
        f.write('Name,ID,Section')

# Only for testing
@app.route('/')
def home():
    if g.user:
        return render_template('HomePage.html', admin=True, mess='Logged in as Administrator', user=session['admin'])

    return render_template('HomePage.html', admin=False, datetoday2=datetoday2)

@app.route('/attendance')
def take_attendance():
    # your logic
    return render_template('Attendance.html', admin=g.user is not None, datetoday=datetoday)

@app.route('/adduser')
def add_user():
    return render_template('AddUser.html', admin=g.user is not None)

@app.route('/attendancelist')
def attendance_list():
    return render_template('AttendanceList.html', admin=g.user is not None, datetoday=datetoday)

@app.route('/registeruserlist')
def register_user_list():
    return render_template('RegisterUserList.html', admin=g.user is not None)

@app.route('/unregisteruserlist')
def unregister_user_list():
    return render_template('UnregisterUserList.html', admin=g.user is not None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '12345':
            session['admin'] = request.form['username']
            return redirect(url_for('home', admin=True, mess='Logged in as Administrator'))
        else:
            return render_template('LogInFrom.html', mess='Incorrect Username or Password')
    return render_template('LogInForm.html')

# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('LogInFrom.html')

# Main Function
if __name__ == '__main__':
    app.run(port=5001, debug=True)