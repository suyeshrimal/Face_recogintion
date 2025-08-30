import os
import csv
import shutil
import time
from datetime import datetime, date

import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, request, session, g, url_for
from dotenv import load_dotenv

from sklearn.neighbors import KNeighborsClassifier
import joblib

from flask_mysqldb import MySQL
import MySQLdb.cursors
from werkzeug.security import generate_password_hash, check_password_hash
# Load environment variables
load_dotenv()

# Initialize Flask app
app=Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Flask error handler
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('Error.html')

# MySQL configurations
app.config['MYSQL_HOST'] = os.getenv("MYSQL_HOST")
app.config['MYSQL_USER'] = os.getenv("MYSQL_USER")
app.config['MYSQL_PASSWORD'] = os.getenv("MYSQL_PASSWORD")
app.config['MYSQL_DB'] = os.getenv("MYSQL_DB")

mysql = MySQL(app)

def load_csv_safely(csv_path, expected_columns):
    """Load a CSV safely. If missing or malformed, create/reset with expected columns."""
    df = pd.DataFrame(columns=expected_columns)

    try:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path)
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                print(f"[Fix] Missing columns {missing_cols} in {csv_path}. Resetting.")
                df = pd.DataFrame(columns=expected_columns)
                df.to_csv(csv_path, index=False)
        else:
            print(f"[Fix] File missing or empty: {csv_path}. Creating new.")
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[Fix] Error reading {csv_path}: {e}. Resetting.")
        df.to_csv(csv_path, index=False)

    return df

# Flask assign admin
@app.before_request
def before_request():
    g.user = None
    if 'admin' in session:
        g.user = session['admin']

# Current Date & Time
datetoday = datetime.today().strftime("%d-%m-%Y")
datetoday2 = datetime.today().strftime("%d %B %Y")

# Capture the video
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
folders = ['Attendance', 'UserList', 'static/faces', 'final_model']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ======= Ensure required CSV files exist ========
files_with_headers = {
    f'Attendance/{datetoday}.csv': 'Name,ID,Section,Time',
    'UserList/Registered.csv': 'Name,ID,Section',
    'UserList/Unregistered.csv': 'Name,ID,Section'
}

for file_path, header in files_with_headers.items():
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w') as f:
            f.write(header + '\n')

# ======= Global Variables =======
# KNN model is loaded on-demand from static/face_recognition_model.pkl

# ======= Remove Empty Rows From CSV Files =======
def remove_empty_cells():
    try:
        csv_paths = ['UserList/Registered.csv', 'UserList/Unregistered.csv']
        if os.path.isdir('Attendance'):
            csv_paths += [os.path.join('Attendance', f) for f in os.listdir('Attendance')]

        for file_path in csv_paths:
            if os.path.isfile(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df = df.dropna(how='any')
                    df = df[df.apply(lambda row: all(str(cell).strip() != '' for cell in row), axis=1)]
                    df.to_csv(file_path, index=False, header=True)
                except pd.errors.EmptyDataError:
                    print(f"[Warning] Empty file skipped: {file_path}")
                except Exception as e:
                    print(f"[Error] Failed processing {file_path}: {e}")

    except Exception as e:
        print(f"[Error] While cleaning CSVs: {e}")
        
# ======= Total Registered Users ========
def totalreg():
    faces_dir = 'static/faces'
    if not os.path.isdir(faces_dir):
        return 0
    return len([name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))])

# ======= Get Face From Image =========
def extract_faces(img):
    try:
        if img is None:
            print("[DEBUG] extract_faces: Input image is None")
            return np.array([])
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray_img, 1.5, 7)
        if len(face_points) > 0:
            print(f"[DEBUG] extract_faces: Detected {len(face_points)} face(s)")
        return face_points
    except Exception as e:
        print(f"[ERROR] extract_faces: Face detection failed - {e}")
        return np.array([])

# ======= Train Model Using Available Faces ========
def train_model():
    print("[INFO] train_model: Starting model training...")
    
    if 'face_recognition_model.pkl' in os.listdir('static'):
        print("[INFO] train_model: Removing old model file...")
        os.remove('static/face_recognition_model.pkl')

    if len(os.listdir('static/faces')) == 0:
        print("[WARNING] train_model: No face images found in static/faces/")
        return

    print(f"[INFO] train_model: Found {len(os.listdir('static/faces'))} user directories")
    
    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    
    for user in user_list:
        user_images = os.listdir(f'static/faces/{user}')
        print(f"[INFO] train_model: Processing user '{user}' with {len(user_images)} images")
        
        for img_name in user_images:
            img_path = f'static/faces/{user}/{img_name}'
            img = cv2.imread(img_path)
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
            else:
                print(f"[WARNING] train_model: Failed to load image {img_path}")

    print(f"[INFO] train_model: Total faces processed: {len(faces)}")
    print(f"[INFO] train_model: Total labels: {len(labels)}")
    
    if len(faces) == 0:
        print("[ERROR] train_model: No valid faces found for training")
        return

    faces = np.array(faces)
    print(f"[INFO] train_model: Training KNN model with {len(faces)} samples...")
    
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(faces, labels)
    
    model_path = 'static/face_recognition_model.pkl'
    joblib.dump(knn, model_path)
    print(f"[SUCCESS] train_model: Model saved to {model_path}")
    print(f"[INFO] train_model: Model trained on {len(set(labels))} unique users")

# ======= Load KNN Model =======
def load_cnn_model():
    # This function is kept for compatibility but KNN model is loaded on-demand
    pass

# ======= Identify Face Using KNN Model ========
def identify_face(face_array):
    try:
        model_path = 'static/face_recognition_model.pkl'
        if not os.path.exists(model_path):
            print(f"[WARNING] identify_face: Model file not found at {model_path}")
            return ["Unknown"]
        
        print(f"[DEBUG] identify_face: Loading model from {model_path}")
        model = joblib.load(model_path)
        
        print(f"[DEBUG] identify_face: Predicting face with shape {face_array.shape}")
        prediction = model.predict(face_array)
        print(f"[DEBUG] identify_face: Prediction result: {prediction}")
        
        return prediction
    except Exception as e:
        print(f"[ERROR] identify_face: Face identification failed - {e}")
        return ["Unknown"]

# ======= Remove Attendance of Deleted User ======
def remAttendance():

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Collect valid IDs from both user tables
    cur.execute("SELECT id FROM student WHERE status='registered'")
    registered_ids = {str(row['id']) for row in cur.fetchall()}

    cur.execute("SELECT id FROM student WHERE status='unregistered'")
    unregistered_ids = {str(row['id']) for row in cur.fetchall()}

    valid_ids = registered_ids | unregistered_ids

    # If there are valid IDs, remove all attendance records that don't belong to them
    if valid_ids:
        ids_str = ",".join([f"'{i}'" for i in valid_ids])
        cur.execute(f"DELETE FROM attendance WHERE id NOT IN ({ids_str})")

    mysql.connection.commit()
    cur.close()

# ======== Get Info From Attendance File =========
def extract_attendance():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    datetoday_mysql = date.today().strftime("%Y-%m-%d")
        
    query = """
        SELECT a.name, a.id, a.section, a.time,
               COALESCE(s.status, 'Unknown') AS status
        FROM attendance a
        LEFT JOIN student s ON a.id = s.id
        WHERE DATE(a.time) = %s
        ORDER BY a.time ASC
    """
    cur.execute(query, (datetoday_mysql,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return [], [], [], [], datetoday, [], 0

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec   = [r['section'] for r in rows]
    times = [r['time'].strftime("%H:%M:%S") for r in rows]
    reg   = [r['status'] for r in rows]
    l     = len(rows)

    return names, rolls, sec, times, datetoday, reg, l

# ======== Save Attendance =========
def add_attendance(name):
    try:
        print(f"[INFO] add_attendance: Processing attendance for '{name}'")
        
        if '$' not in name:
            print(f"[ERROR] add_attendance: Invalid name format: {name}")
            return
            
        parts = name.split('$')
        if len(parts) < 3:
            print(f"[ERROR] add_attendance: Insufficient name parts: {name}")
            return
            
        username = parts[0]
        userid = parts[1]
        usersection = parts[2]
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[INFO] add_attendance: User: {username}, ID: {userid}, Section: {usersection}")
        print(f"[INFO] add_attendance: Current time: {current_datetime}")

        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        # Check if already marked today (ignoring time, only by DATE)
        print(f"[DEBUG] add_attendance: Checking if user {userid} already marked attendance today")
        cur.execute("""
            SELECT * FROM attendance 
            WHERE id=%s AND DATE(time)=%s
        """, (userid, datetime.now().strftime("%Y-%m-%d")))
        already = cur.fetchone()

        if already:
            print(f"[INFO] add_attendance: User {userid} already marked attendance today")
            cur.close()
            return 

        # Insert new attendance with full date+time
        print(f"[INFO] add_attendance: Inserting new attendance record for user {userid}")
        cur.execute("""
            INSERT INTO attendance (id, name, section, time)
            VALUES (%s, %s, %s, %s)
        """, (userid, username, usersection, current_datetime))
        mysql.connection.commit()
        cur.close()
        print(f"[SUCCESS] add_attendance: Attendance recorded successfully for user {userid}")
    except Exception as e:
        print(f"[ERROR] add_attendance: Failed to add attendance - {e}")

# ======= Flask Home Page =========
@app.route('/')
def home():
    if g.user:
        return render_template('HomePage.html', admin=True, mess='Logged in as Administrator', user=session['admin'])
    return render_template('HomePage.html', admin=False, datetoday2=datetoday2)

# ======= Flask Attendance Page =========
@app.route('/attendance')
def take_attendance():
    # Fetch today's attendance from MySQL
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    
    return render_template(
        'Attendance.html',
        names=names,
        rolls=rolls,
        sec=sec,
        times=times,
        l=l,
        datetoday2=datetoday2
    )

@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    print("[INFO] attendancebtn: Starting attendance capture...")
    
    if len(os.listdir('static/faces')) == 0:
        print("[WARNING] attendancebtn: No face images found in database")
        return render_template('Attendance.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

    print(f"[INFO] attendancebtn: Found {len(os.listdir('static/faces'))} users in database")
    
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print("[INFO] attendancebtn: Model not found, training new model...")
        train_model()
    else:
        print("[INFO] attendancebtn: Using existing trained model")

    print("[INFO] attendancebtn: Opening camera...")
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("[ERROR] attendancebtn: Failed to open camera")
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    print("[SUCCESS] attendancebtn: Camera opened successfully")
    print("[INFO] attendancebtn: Starting face detection loop...")
    
    ret = True
    j = 1
    flag = -1
    frame_count = 0
    
    while ret:
        ret, frame = cap.read()
        frame_count += 1
        
        if frame_count % 30 == 0:  # Log every 30 frames (about 1 second)
            print(f"[DEBUG] attendancebtn: Processing frame {frame_count}")
        
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            
            try:
                print(f"[DEBUG] attendancebtn: Attempting to identify face in frame {frame_count}")
                identified_person = identify_face(face.reshape(1, -1))[0]
                
                if '$' in identified_person:
                    identified_person_name = identified_person.split('$')[0]
                    identified_person_id = identified_person.split('$')[1]
                    print(f"[INFO] attendancebtn: Identified person: {identified_person_name} (ID: {identified_person_id})")
                else:
                    identified_person_name = "Unknown"
                    identified_person_id = "N/A"
                    print(f"[WARNING] attendancebtn: Unknown person identified")

                if flag != identified_person:
                    j = 1
                    flag = identified_person
                    print(f"[INFO] attendancebtn: New person detected, resetting counter")

                if j % 20 == 0:
                    print(f"[INFO] attendancebtn: Marking attendance for {identified_person_name}")
                    add_attendance(identified_person)
            except Exception as e:
                print(f"[ERROR] attendancebtn: Error processing identified person: {e}")
                identified_person_name = "Error"
                identified_person_id = "N/A"

            cv2.putText(frame, f'Name: {identified_person_name}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20),
                        2,
                        cv2.LINE_AA)
            cv2.putText(frame, f'ID: {identified_person_id}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, 'Press Space to close', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255), 2,
                        cv2.LINE_AA)
            j += 1
        else:
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[DEBUG] attendancebtn: No faces detected in frame {frame_count}")
            j = 1
            flag = -1

        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
        cv2.imshow('Attendance', frame)
        # Close with Space instead of Esc per your preference
        if cv2.waitKey(1) == 32:
            print("[INFO] attendancebtn: Space key pressed, closing camera...")
            break

    print("[INFO] attendancebtn: Releasing camera resources...")
    cap.release()
    cv2.destroyAllWindows()
    
    print("[INFO] attendancebtn: Fetching updated attendance data...")
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    print(f"[INFO] attendancebtn: Found {l} attendance records")
    
    return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)

@app.route('/adduser')
def add_user():
    return render_template('AddUser.html')

@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newusersection = request.form['newusersection']
    
    print(f"[INFO] adduserbtn: Starting user registration for {newusername} (ID: {newuserid}, Section: {newusersection})")

    # Open camera
    print("[INFO] adduserbtn: Opening camera for face capture...")
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("[ERROR] adduserbtn: Failed to open camera")
        return render_template('AddUser.html', mess='Camera not available.')

    print("[SUCCESS] adduserbtn: Camera opened successfully")

    # Create user folder for storing images
    userimagefolder = f'static/faces/{newusername}${newuserid}${newusersection}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
        print(f"[INFO] adduserbtn: Created user folder: {userimagefolder}")

    # Check if user already exists in DB
    print(f"[INFO] adduserbtn: Checking if user {newuserid} already exists in database...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE id = %s", (newuserid,))
    existing_user = cur.fetchone()
    if existing_user:
        print(f"[WARNING] adduserbtn: User {newuserid} already exists in database")
        cap.release()
        cur.close()
        return render_template('AddUser.html', mess='User already exists in database.')

    print(f"[INFO] adduserbtn: User {newuserid} is new, proceeding with face capture")

    images_captured = 0
    max_images = 100
    frame_count = 0

    print(f"[INFO] adduserbtn: Starting face capture loop (target: {max_images} images)")

    while images_captured < max_images:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print(f"[WARNING] adduserbtn: Failed to read frame {frame_count}")
            break
            
        faces = extract_faces(frame)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = cv2.resize(frame[y:y+h, x:x+w], (50,50))
                image_path = os.path.join(userimagefolder, f'{images_captured}.jpg')
                cv2.imwrite(image_path, face_img)
                images_captured += 1
                
                if images_captured % 10 == 0:  # Log every 10 images
                    print(f"[INFO] adduserbtn: Captured {images_captured}/{max_images} images")

                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,20), 2)
                
                # Add delay to slow down capture process (0.2 seconds = 20 seconds for 100 images)
                time.sleep(0.2)
        else:
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[DEBUG] adduserbtn: No faces detected in frame {frame_count}")
                
        cv2.putText(frame, f'Images Captured: {images_captured}/{max_images}', (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,20), 2)
        cv2.putText(frame, 'Capturing slowly - 20 secs for 100 images', (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Calculate and display estimated time remaining
        if images_captured > 0:
            remaining_images = max_images - images_captured
            estimated_seconds = remaining_images * 0.2  # 0.2 seconds per image
            cv2.putText(frame, f'Time remaining: ~{estimated_seconds:.1f} seconds', (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
        
        cv2.namedWindow("Collecting Face Data", cv2.WINDOW_NORMAL)
        cv2.imshow("Collecting Face Data", frame)
        # Close with Space instead of Esc per your preference
        if cv2.waitKey(1) == 32:
            print("[INFO] adduserbtn: Space key pressed, stopping capture...")
            break

    print(f"[INFO] adduserbtn: Face capture completed. Total images: {images_captured}")
    cap.release()
    cv2.destroyAllWindows()

    if images_captured == 0:
        print("[ERROR] adduserbtn: No images captured, cleaning up...")
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture valid face images.')

    print(f"[SUCCESS] adduserbtn: Successfully captured {images_captured} face images")

    # Insert new user into MySQL
    print(f"[INFO] adduserbtn: Inserting user {newuserid} into database...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        INSERT INTO student (name, id, section, status)
        VALUES (%s, %s, %s, 'unregistered')
    """, (newusername, newuserid, newusersection))
    mysql.connection.commit()
    cur.close()
    print(f"[SUCCESS] adduserbtn: User {newuserid} inserted into database")

    # Retrain model immediately with new user
    print("[INFO] adduserbtn: Retraining model with new user...")
    train_model()

    # Fetch updated unregistered students
    print("[INFO] adduserbtn: Fetching updated unregistered students list...")
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [str(row['id']) for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)
    
    print(f"[INFO] adduserbtn: Found {l} unregistered students")
    print(f"[SUCCESS] adduserbtn: User registration completed successfully")

    return render_template('UnregisterUserList.html',
                           names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

@app.route('/attendancelist')
def attendance_list():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()
    names, rolls, sec, times, dates, reg, l = extract_attendance()

    return render_template(
        'AttendanceList.html',
        names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg, l=l
    )
    
# ========== Flask Search Attendance by Date ============
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('LogInForm.html')

    date_selected = request.form['date']  # "YYYY-MM-DD"

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        SELECT a.name, a.id, a.section, a.time,
               COALESCE(s.status, 'Unknown') AS status
        FROM attendance a
        LEFT JOIN student s ON a.id = s.id
        WHERE DATE(a.time) = %s
        ORDER BY a.time ASC
    """, (date_selected,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return render_template('AttendanceList.html', names=[], rolls=[], sec=[], times=[], reg=[], l=0,
                               mess="No records for this date.")

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec = [r['section'] for r in rows]
    times = [r['time'].strftime("%H:%M:%S") for r in rows]
    dates = [row['time'] for row in rows]
    reg = [r['status'] for r in rows]
    l = len(rows)

    return render_template('AttendanceList.html',
                        names=names, rolls=rolls, sec=sec,
                        times=times, dates=dates, reg=reg,
                        l=l, mess=f"Total Attendance: {l}")

# ========== Flask Search Attendance by ID ============
@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('LogInForm.html')

    student_id = request.form.get('id')
    if not student_id:
        return render_template('AttendanceList.html', names=[], rolls=[], sec=[], times=[], dates=[], reg=[], l=0, mess="No ID provided!")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM attendance WHERE id = %s", (student_id,))
    rows = cur.fetchall()
    cur.close()

    if rows:
        names = [row['name'] for row in rows]
        rolls = [row['id'] for row in rows]
        sec   = [row['section'] for row in rows]
        times = [row['time'].strftime("%H:%M:%S") if row['time'] else "N/A" for row in rows]
        dates = [row['time'].strftime("%Y-%m-%d") if row['time'] else "N/A" for row in rows]
        reg   = ['Registered' if row['id'] in [r['id'] for r in rows] else 'Unregistered' for row in rows]
        l = len(rows)
        return render_template('AttendanceList.html',
                               names=names, rolls=rolls, sec=sec,
                               times=times, dates=dates, reg=reg,
                               l=l, mess=f"Total Attendance: {l}")
    else:
        return render_template('AttendanceList.html',
                               names=[], rolls=[], sec=[],
                               times=[], dates=[], reg=[],
                               l=0, mess="Nothing Found!")

# ========== Flask Register User List ============
@app.route('/registeruserlist')
def register_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l else "Database is empty!"
    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)

# ========== Flask Unregister a User ============
@app.route('/unregisteruser', methods=['POST'])
def unregisteruser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index (not a number or missing)", 400

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # Get only registered students
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    registered = cur.fetchall()

    if idx < 0 or idx >= len(registered):
        return "Invalid index", 400

    user = registered[idx]
    userid, username, section = user['id'], user['name'], user['section']

        # Move the face folder (optional)
    old_folder = f"static/faces/{username}${userid}${section}"
    new_folder = f"static/faces/{username}${userid}$None"
    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)
        
    # Update status in single student table
    cur.execute(
        "UPDATE student SET status='unregistered', section=NULL WHERE id=%s",
        (userid,)
    )
    mysql.connection.commit()
    cur.close()

    # Return updated list of registered students
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec = [r['section'] for r in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l > 0 else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)

# ========== Flask Delete a User from Unregistered List ============
@app.route('/deleteunregistereduser', methods=['POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # Fetch unregistered students only
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    unregistered = cur.fetchall()

    if idx >= len(unregistered):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    user = unregistered[idx]
    username, userid, usersec = user['name'], user['id'], user['section']

    folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    # Delete student from table
    cur.execute("DELETE FROM student WHERE id = %s AND status='unregistered'", (userid,))
    mysql.connection.commit()
    cur.close()

    return redirect(url_for('unregister_user_list'))
        
# ========== Flask Unregister User List ============
@app.route('/unregisteruserlist')
def unregister_user_list():
    if not g.user:
        return render_template('LogInForm.html')
    
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Database is empty!")

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

# ========== Flask Register a User ============
@app.route('/registeruser', methods=['POST'])
def registeruser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
        section = request.form['section']
    except (ValueError, KeyError):
        return "Invalid input", 400

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # Get all unregistered students
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    unregistered = cur.fetchall()

    if idx < 0 or idx >= len(unregistered):
        return "Invalid user index", 400

    user = unregistered[idx]
    name, userid = user['name'], user['id']

    # Move the face folder
    old_folder = f"static/faces/{name}${userid}$None"
    new_folder = f"static/faces/{name}${userid}${section}"
    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)

    # Update status and section in single student table
    cur.execute(
        "UPDATE student SET status='registered', section=%s WHERE id=%s",
        (section, userid)
    )
    mysql.connection.commit()
    cur.close()

    # Reload unregistered list
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [r['name'] for r in rows]
    rolls = [r['user_id'] for r in rows]  # or r['id'] if you prefer DB ID
    secs = [r['section'] for r in rows]
    l = len(rows)

    mess = f'Number of Unregistered Students: {l}' if l > 0 else "Database is empty!"
    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=secs, l=l, mess=mess)

# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    registered = cur.fetchall()

    if idx >= len(registered):
        return render_template('RegisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    user = registered[idx]
    username, userid, usersec = user['name'], user['id'], user['section']

    # Delete face folder
    folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    # Delete from DB
    cur.execute("DELETE FROM student WHERE id = %s and status='registered'", (userid,))
    mysql.connection.commit()
    cur.close()

    # Refresh list
    return redirect(url_for('register_user_list'))
    
# ======== Flask Login =========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        admin_id = request.form['admin_id']
        password = request.form['password']

        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        # Step 1: Fetch user
        cur.execute("SELECT * FROM admin_signup WHERE admin_id = %s", (admin_id,))
        user = cur.fetchone()

        if user and check_password_hash(user['password'], password):
            # Step 2: Insert login record (store login time, not password!)
            try:
                cur.execute("INSERT INTO admin_login (admin_id, username) VALUES (%s, %s)",
                            (admin_id, user['username']))
                mysql.connection.commit()
                print("Stored login record for:", admin_id)
            except Exception as e:
                mysql.connection.rollback()
                print("!!!Error inserting login record:", e)

            # Step 3: Save session
            session['admin'] = admin_id
            cur.close()
            return redirect(url_for('home', admin=True, mess=f'Logged in as {admin_id}'))
        else:
            cur.close()
            return render_template('LogInForm.html', mess='Incorrect Admin ID or Password')

    return render_template('LogInForm.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('LogInForm.html')

# ======== Flask Sign Up =========
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    
    if request.method == 'POST':
        admin_id = request.form['admin_id']
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM admin_signup WHERE admin_id = %s", (admin_id,))
        existing_user = cur.fetchone()

        # Check if user already exists
        if existing_user:
            mess = "Admin ID already exists!"
            return render_template('SignUpPage.html', mess=mess)

        try:
            # Insert new user
            cur.execute("INSERT INTO admin_signup (admin_id, username, password) VALUES (%s, %s, %s)",
                        (admin_id, username, hashed_password))
            mysql.connection.commit()
            mess = "Account created successfully! Please log in."
            return redirect(url_for('login', mess=mess))
        except Exception as e:
            mysql.connection.rollback()
            mess = f"Database error: {str(e)}"
            return render_template('SignUpPage.html', mess=mess)
        finally:
            cur.close()

    return render_template('SignUpPage.html')

@app.route('/adminlog')
def adminlog():
    # Ensure admin is logged in
    if 'admin' not in session:
        return render_template('LogInForm.html', mess="Please log in first.")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Fetch all admin users from signup table
    cur.execute("""
        SELECT l.admin_id, s.username 
        FROM admin_login l
        JOIN admin_signup s ON l.admin_id = s.admin_id
        ORDER BY l.admin_id DESC
    """)
    logs = cur.fetchall()
    cur.close()

    admin_ids = [log['admin_id'] for log in logs]
    usernames = [log['username'] for log in logs]

    return render_template('AdminLog.html', admin_ids=admin_ids, usernames=usernames, l=len(logs))

# Main Function
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Face Recognition Attendance System")
    print("=" * 60)
    print(f"[INFO] Main: Current date: {datetoday}")
    print(f"[INFO] Main: Current date (formatted): {datetoday2}")
    print(f"[INFO] Main: Face detector loaded: {face_detector is not None}")
    print(f"[INFO] Main: Static faces directory: {'static/faces' in os.listdir('.')}")
    print(f"[INFO] Main: KNN model will be trained on-demand when needed")
    print("=" * 60)
    
    # KNN model is trained on-demand when needed
    app.run(port=5001, debug=True)