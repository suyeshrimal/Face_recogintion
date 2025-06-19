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


# Only for testing
@app.route('/')
def home():
    return render_template('HomePage.html', date=datetoday2)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        pass
    return render_template('LogInForm.html')

# Main Function
if __name__ == '__main__':
    app.run(debug=True)