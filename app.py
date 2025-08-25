import os
import csv
import shutil
from datetime import datetime, date

import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, request, session, g, url_for
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from flask_mysqldb import MySQL
import MySQLdb.cursors
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
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
eye_detector = cv2.CascadeClassifier('static/haarcascade_eye.xml')
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
cnn_model = None
class_names = {}

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
def extract_faces_and_eyes(img):
    if img is None or img.size == 0:
        return (), ()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    eyes_list = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        eyes_global = [(ex + x, ey + y, ew, eh) for (ex, ey, ew, eh) in eyes]
        eyes_list.append(eyes_global)

    return faces, eyes_list

# ======= Train Model Using Available Faces ========
def train_model():
    global cnn_model, class_names

    face_dir = 'static/faces'
    model_path = 'final_model/face_recognition_model.h5'
    class_path = 'final_model/class_names.pkl'

    if not os.path.exists(face_dir) or len(os.listdir(face_dir)) == 0:
        print("[Info] No faces to train on.")
        return
        
    # Remove old model if exists
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(class_path):
        os.remove(class_path)

    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        validation_split=0.2
    )

    # Train and validation generators
    train_data = datagen.flow_from_directory(
        face_dir, target_size=(224,224), batch_size=32,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_data = datagen.flow_from_directory(
        face_dir, target_size=(224,224), batch_size=32,
        class_mode='categorical', subset='validation', shuffle=True
    )
    # Build CNN model
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
    ])

    cnn_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"]
    )

    # Train the model
    print("[INFO] Training model...")
    cnn_model.fit(
        train_data, steps_per_epoch=len(train_data),
        validation_data=val_data, validation_steps=len(val_data),
        epochs=20, verbose=1
    )

    # Save model and class names
    os.makedirs('final_model', exist_ok=True)
    cnn_model.save(model_path)
    
    # Create class names mapping from training data
    class_names = {v: k for k, v in train_data.class_indices.items()}
    with open(class_path, 'wb') as f:
        pickle.dump(class_names, f)

    print("Model trained and saved successfully!")
    print(f"Classes: {class_names}")

# ======= Load CNN Model =======
def load_cnn_model():
    global cnn_model, class_names

    model_path = 'final_model/face_recognition_model.h5'
    class_names_path = 'final_model/class_names.pkl'

    # Load CNN model
    if os.path.exists(model_path):
        try:
            cnn_model = load_model(model_path)
        except Exception as e:
            cnn_model = None
    else:
        print(f"[WARNING] CNN model not found at {model_path}.")
        cnn_model = None

    # Load class_names
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path, 'rb') as f:
                class_names = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load class names: {e}")
            class_names = {}
    else:
        print(f"[WARNING] class_names.pkl not found.")
        class_names = {}

    # Validate saved classes vs current face folders to avoid stale labels influencing predictions
    try:
        faces_root = 'static/faces'
        if os.path.isdir(faces_root):
            current_dirs = {d for d in os.listdir(faces_root) if os.path.isdir(os.path.join(faces_root, d))}
            saved_labels = set(class_names.values()) if isinstance(class_names, dict) else set()
            if saved_labels and not saved_labels.issubset(current_dirs):
                print("[INFO] Detected class mismatch with current folders. Retraining model...")
                train_model()
                if os.path.exists(model_path):
                    cnn_model = load_model(model_path)
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'rb') as f:
                        class_names = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Class validation failed: {e}")

# ======= Identify Face Using CNN Model ========
def identify_face(face_img):
    global cnn_model, class_names
    
    if face_img is None or cnn_model is None:
        print("[DEBUG] No model or image available")
        return "Unknown"

    try:
        # Preprocess face image
        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict and get class with confidence
        preds = cnn_model.predict(face_input, verbose=0)[0]
        top1 = int(np.argmax(preds))
        top2 = int(np.argsort(preds)[-2]) if preds.size >= 2 else top1
        p1 = float(preds[top1])
        p2 = float(preds[top2])

        # Confidence floor and separation margin vs runner-up to avoid near-tie flips
        CONFIDENCE_THRESHOLD = 0.7
        MARGIN = 0.15

        if p1 < CONFIDENCE_THRESHOLD or (p1 - p2) < MARGIN:
            return "Unknown"

        # Map to class name
        return class_names.get(top1, "Unknown")
    
    except Exception as e:
        print(f"[Error] Face identification failed: {e}")
        return "Unknown"

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
    times = [r['time'].strftime("%H:%M:%S") for r in rows]  # just time
    reg   = [r['status'] for r in rows]
    l     = len(rows)

    return names, rolls, sec, times, datetoday, reg, l

# ======== Save Attendance =========
def add_attendance(name):
    username, userid, usersection = name.split('$')
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Check if already marked today (ignoring time, only by DATE)
    cur.execute("""
        SELECT * FROM attendance 
        WHERE id=%s AND DATE(time)=%s
    """, (userid, datetime.now().strftime("%Y-%m-%d")))
    already = cur.fetchone()

    if already:
        cur.close()
        return 

    # Insert new attendance with full date+time
    cur.execute("""
        INSERT INTO attendance (id, name, section, time)
        VALUES (%s, %s, %s, %s)
    """, (userid, username, usersection, current_datetime))
    mysql.connection.commit()
    cur.close()

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
    global cnn_model, class_names
    
    if len(os.listdir('static/faces')) == 0:
        return render_template('Attendance.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

    # Ensure model exists and is loaded
    if cnn_model is None:
        print("[INFO] Model not loaded, attempting to load...")
        load_cnn_model()
        
    if cnn_model is None:
        print("[INFO] No model found, training new model...")
        train_model()
        load_cnn_model()

    if cnn_model is None:
        return render_template('Attendance.html', datetoday2=datetoday2,
                               mess='Failed to load or train model.')

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    ret = True
    # temporal smoothing & lock-on
    consecutive_counts = {}
    current_lock = None
    lock_grace = 10  # frames to keep lock if momentarily lost
    lock_timer = 0
    NEED_CONSEC = 5   # frames required to confirm identity

    while ret:
        ret, frame = cap.read()
        faces, eyes_list = extract_faces_and_eyes(frame)

        identified_person_name = "Unknown"
        identified_person_id = "N/A"

        if faces is not None and len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                if w < 100 or h < 100:  # Skip very small faces
                    continue
                    
                if i < len(eyes_list) and len(eyes_list[i]) > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    for (ex, ey, ew, eh) in eyes_list[i]:
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    face_img = cv2.resize(frame[y:y + h, x:x + w], (224, 224))
                    identified_person = identify_face(face_img)

                    # If we already locked an identity, keep it as long as lock_timer remains
                    if current_lock is not None and '$' in current_lock:
                        identified_person_name, identified_person_id, *_ = current_lock.split('$')
                        lock_timer = lock_grace
                    else:
                        # Build up consecutive evidence before locking and marking attendance
                        if identified_person is not None and '$' in identified_person:
                            consecutive_counts[identified_person] = consecutive_counts.get(identified_person, 0) + 1
                            if consecutive_counts[identified_person] >= NEED_CONSEC:
                                current_lock = identified_person
                                consecutive_counts.clear()
                                add_attendance(current_lock)
                                identified_person_name, identified_person_id, *_ = current_lock.split('$')
                        else:
                            consecutive_counts.clear()

                    cv2.putText(frame, f'Name: {identified_person_name}', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'ID: {identified_person_id}', (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'Press Esc to close', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255), 2, cv2.LINE_AA)

        else:
            # if no faces, decay the lock
            if current_lock is not None:
                lock_timer -= 1
                if lock_timer <= 0:
                    current_lock = None

        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance', 800, 600)
        cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, sec, times, dates, reg, l = extract_attendance()
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

    # Open camera
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return render_template('AddUser.html', mess='Camera not available.')

    # Create user folder for storing images
    userimagefolder = f'static/faces/{newusername}${newuserid}${newusersection}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    # Check if user already exists in DB
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE id = %s", (newuserid,))
    existing_user = cur.fetchone()
    if existing_user:
        cap.release()
        cur.close()
        return render_template('AddUser.html', mess='User already exists in database.')

    images_captured = 0
    max_images = 100

    while images_captured < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        faces, eyes_list = extract_faces_and_eyes(frame)
        if faces is not None:
            for i, (x, y, w, h) in enumerate(faces):
                if i < len(eyes_list) and len(eyes_list[i]) > 0:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (224,224))
                    cv2.imwrite(
                        os.path.join(userimagefolder, f'{images_captured}.jpg'),
                        face_img
                    )
                    images_captured += 1

                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,20), 2)
                    for (ex, ey, ew, eh) in eyes_list[i]:
                        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.putText(frame, f'Images Captured: {images_captured}/{max_images}', (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,20), 2)
        cv2.namedWindow("Collecting Face Data", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Collecting Face Data", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Collecting Face Data", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if images_captured == 0:
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture valid face images.')

    # Insert new user into MySQL
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        INSERT INTO student (name, id, section, status)
        VALUES (%s, %s, %s, 'unregistered')
    """, (newusername, newuserid, newusersection))
    mysql.connection.commit()
    cur.close()

    # Retrain model immediately with new user
    train_model()
    load_cnn_model()  # Reload the model after training

    # Fetch updated unregistered students
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [str(row['id']) for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

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
    # Load or train model
    load_cnn_model()
    
    if cnn_model is None:
        print("[INFO] No model found, will train when needed.")
    app.run(port=5001, debug=True)