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
    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    valid_ids = set(map(str, dfr['ID'])) | set(map(str, dfu['ID']))

    attendance_dir = 'Attendance'
    if not os.path.isdir(attendance_dir):
        return

    for file_name in os.listdir(attendance_dir):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(attendance_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            df_filtered = df[df['ID'].astype(str).isin(valid_ids)]
            df_filtered.to_csv(file_path, index=False)
        except Exception as e:
            print(f"[Warning] Could not process {file_path}: {e}")

    remove_empty_cells()

# ======== Get Info From Attendance File =========
def extract_attendance():
    today_str = date.today().strftime("%Y-%m-%d")
    attendance_file = os.path.join('Attendance', f'{today_str}.csv')

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    if not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0:
        return [], [], [], [], today_str, [], 0

    df = pd.read_csv(attendance_file)

    names = df['Name'].tolist()
    rolls = df['ID'].astype(str).tolist()
    sec = df['Section'].tolist()
    times = df['Time'].tolist()
    dates = [today_str] * len(df)

    unreg_ids = set(map(str, dfu['ID']))
    reg_ids = set(map(str, dfr['ID']))

    reg = ["Unregistered" if uid in unreg_ids else "Registered" if uid in reg_ids else "Unknown" for uid in rolls]

    return names, rolls, sec, times, dates, reg, len(df)

# ======== Save Attendance =========
def add_attendance(name):
    username, userid, usersection = name.split('$')
    current_time = datetime.now().strftime("%I:%M %p")
    
    file_path = f'Attendance/{datetoday}.csv'
    
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        df = pd.DataFrame(columns=['Name', 'ID', 'Section', 'Time'])
        df.to_csv(file_path, index=False)
    
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'ID' not in df.columns or 'Name' not in df.columns:
            df = pd.DataFrame(columns=['Name', 'ID', 'Section', 'Time'])
            df.to_csv(file_path, index=False)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['Name', 'ID', 'Section', 'Time'])
        df.to_csv(file_path, index=False)

    user_rows = df[df['ID'].astype(str) == str(userid)]
    
    if user_rows.empty:
        new_entry = pd.DataFrame([[username, userid, usersection, current_time]],
                                 columns=['Name', 'ID', 'Section', 'Time'])
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        try:
            last_time = user_rows.iloc[-1]['Time']
            start_time = datetime.strptime(last_time, "%I:%M %p")
            end_time = datetime.strptime(current_time, "%I:%M %p")
            delta = (end_time - start_time).total_seconds() / 60
            if delta > 40:
                new_entry = pd.DataFrame([[username, userid, usersection, current_time]],
                                         columns=['Name', 'ID', 'Section', 'Time'])
                new_entry.to_csv(file_path, mode='a', index=False, header=False)
        except Exception:
            new_entry = pd.DataFrame([[username, userid, usersection, current_time]],
                                     columns=['Name', 'ID', 'Section', 'Time'])
            new_entry.to_csv(file_path, mode='a', index=False, header=False)

# ======= Flask Home Page =========
@app.route('/')
def home():
    if g.user:
        return render_template('HomePage.html', admin=True, mess='Logged in as Administrator', user=session['admin'])
    return render_template('HomePage.html', admin=False, datetoday2=datetoday2)

# ======= Flask Attendance Page =========
@app.route('/attendance')
def take_attendance():
    attendance_file = f'Attendance/{datetoday}.csv'

    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,ID,Section,Time')

    remove_empty_cells()
    names, rolls, sec, times, dates, reg, l = extract_attendance()

    return render_template(
        'Attendance.html',
        names=names, rolls=rolls, sec=sec, times=times, l=l, datetoday2=datetoday2
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('AddUser.html', mess='Camera not available.')

    userimagefolder = f'static/faces/{newusername}${newuserid}${newusersection}'
    os.makedirs(userimagefolder, exist_ok=True)

    csv_unreg_path = os.path.join(app.root_path, 'UserList', 'Unregistered.csv')
    csv_reg_path = os.path.join(app.root_path, 'UserList', 'Registered.csv')

    dfu = load_csv_safely(csv_unreg_path, ['Name', 'ID', 'Section'])
    dfr = load_csv_safely(csv_reg_path, ['Name', 'ID', 'Section'])

    if str(newuserid) in map(str, dfu['ID']) or str(newuserid) in map(str, dfr['ID']):
        cap.release()
        return render_template('AddUser.html', mess='User already exists.')

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
        cv2.imshow('Collecting Face Data', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if images_captured == 0:
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture valid face images.')

    with open(csv_unreg_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([newusername, newuserid, newusersection])

    # Retrain model immediately with new user
    train_model()
    load_cnn_model()  # Reload the model after training

    dfu = load_csv_safely(csv_unreg_path, ['Name', 'ID', 'Section'])
    names = dfu['Name'].tolist()
    rolls = dfu['ID'].astype(str).tolist()
    sec = dfu['Section'].tolist()
    l = len(dfu)

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
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
    
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('LogInForm.html')

    input_date = request.form['date']
    try:
        year, month, day = input_date.split('-')
        attendance_file = f'Attendance/{day}-{month}-{year}.csv'
    except Exception as e:
        return render_template('AttendanceList.html', mess="Invalid date format!")

    if not os.path.exists(attendance_file):
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template(
            'AttendanceList.html',
            names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg, l=0,
            mess="Nothing Found!"
        )

    names, rolls, sec, times, dates, reg = [], [], [], [], [], []
    l = 0

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    with open(attendance_file, "r") as f:
        csv_file = csv.reader(f, delimiter=",")
        next(csv_file, None)

        for row in csv_file:
            names.append(row[0])
            rolls.append(row[1])
            sec.append(row[2])
            times.append(row[3])
            dates.append(f'{day}-{month}-{year}')

            if str(row[1]) in map(str, dfu['ID']):
                reg.append("Unregistered")
            elif str(row[1]) in map(str, dfr['ID']):
                reg.append("Registered")
            else:
                reg.append("Unknown")

            l += 1

    return render_template(
        'AttendanceList.html',
        names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg, l=l,
        totalreg=totalreg(), mess=None if l else "Nothing Found!"
    )

@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('LogInForm.html')

    search_id = request.form.get('id')
    if not search_id:
        return render_template('AttendanceList.html', mess="No ID provided!")

    names, rolls, sec, times, dates, reg = [], [], [], [], [], []
    count = 0

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    attendance_dir = 'Attendance'
    if not os.path.exists(attendance_dir):
        return render_template('AttendanceList.html', mess="Attendance directory not found!")

    for filename in os.listdir(attendance_dir):
        filepath = os.path.join(attendance_dir, filename)
        if not filename.endswith('.csv') or not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "r") as f:
                csv_file = csv.reader(f, delimiter=",")
                next(csv_file, None)

                for row in csv_file:
                    if len(row) < 4:
                        continue

                    if row[1] != search_id:
                        continue

                    names.append(row[0])
                    rolls.append(row[1])
                    sec.append(row[2])
                    times.append(row[3])
                    dates.append(filename.replace('.csv', ''))

                    if row[1] in map(str, dfu['ID']):
                        reg.append("Unregistered")
                    elif row[1] in map(str, dfr['ID']):
                        reg.append("Registered")
                    else:
                        reg.append("Unknown")

                    count += 1

        except Exception as e:
            print(f"[Warning] Skipping file {filename} due to error: {e}")

    return render_template(
        'AttendanceList.html',
        names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg, l=count,
        mess=f'Total Attendance: {count}' if count else "Nothing Found!"
    )

@app.route('/registeruserlist')
def register_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()
    csv_path = 'UserList/Registered.csv'
    names, rolls, sec = [], [], []

    try:
        with open(csv_path, "r") as f:
            csv_file = csv.reader(f, delimiter=",")
            headers = next(csv_file, None)
            for row in csv_file:
                if len(row) != 3:
                    print(f"[Warning] Skipping malformed row: {row}")
                    continue
                names.append(row[0].strip())
                rolls.append(row[1].strip())
                sec.append(row[2].strip())
    except FileNotFoundError:
        print(f"[Error] {csv_path} not found.")
    except Exception as e:
        print(f"[Error] Reading {csv_path}: {e}")

    total = len(names)
    mess = f'Number of Registered Students: {total}' if total else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=total, mess=mess)

@app.route('/unregisteruser', methods=['POST'])
def unregisteruser():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()

    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index (not a number or missing)", 400

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    if idx < 0 or idx >= len(dfr):
        return f"Invalid index: {idx} for {len(dfr)} users", 400

    row = dfr.iloc[idx].copy()

    old_folder = f"static/faces/{row['Name']}${row['ID']}${row['Section']}"
    new_folder = f"static/faces/{row['Name']}${row['ID']}$None"

    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)

    row['Section'] = 'None'

    dfu = pd.concat([dfu, pd.DataFrame([row])], ignore_index=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    dfr = dfr.drop(dfr.index[idx])
    dfr.to_csv('UserList/Registered.csv', index=False)

    load_cnn_model()
    remove_empty_cells()

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    names = dfr['Name'].tolist()
    rolls = dfr['ID'].astype(str).tolist()
    sec = dfr['Section'].tolist()
    l = len(dfr)
    mess = f'Number of Registered Students: {l}' if l else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)

@app.route('/deleteregistereduser', methods=['GET', 'POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index provided", 400

    remove_empty_cells()
    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])

    if idx < 0 or idx >= len(dfr):
        return f"Invalid index: {idx} for {len(dfr)} users", 400

    username, userid, usersec = dfr.iloc[idx][['Name','ID','Section']]

    face_folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(face_folder):
        shutil.rmtree(face_folder)
        train_model()
        load_cnn_model()

    dfr = dfr.drop(dfr.index[idx])
    dfr.to_csv('UserList/Registered.csv', index=False)

    remove_empty_cells()
    remAttendance()

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    names = dfr['Name'].tolist()
    rolls = dfr['ID'].astype(str).tolist()
    sec = dfr['Section'].tolist()
    l = len(dfr)
    mess = f'Number of Registered Students: {l}' if l else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)
        
@app.route('/unregisteruserlist')
def unregister_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()
    df = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    if df.empty:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0,
                               mess="Database is empty!")

    names = df['Name'].astype(str).tolist()
    rolls = df['ID'].astype(str).tolist()
    sec = df['Section'].astype(str).tolist()
    l = len(df)
    
    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
        new_section = request.form['section'].strip()
    except (KeyError, ValueError):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0,
                               mess="Invalid input.")

    remove_empty_cells()

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    if idx < 0 or idx >= len(dfu):
        return render_template('UnregisterUserList.html', names=[], rolls=rolls, sec=sec, l=l, mess="Invalid user index.")

    row = dfu.iloc[idx].copy()
    name, uid = row['Name'], row['ID']

    old_folder = f'static/faces/{name}${uid}$None'
    new_folder = f'static/faces/{name}${uid}${new_section}'
    if os.path.exists(old_folder):
        shutil.move(old_folder, new_folder)

    row['Section'] = new_section
    dfr = pd.concat([dfr, pd.DataFrame([row])], ignore_index=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    dfu = dfu.drop(index=idx)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    load_cnn_model()
    remove_empty_cells()

    names = dfu['Name'].tolist()
    rolls = dfu['ID'].astype(str).tolist()
    secs = dfu['Section'].tolist()
    l = len(dfu)
    mess = f'Number of Unregistered Students: {l}' if l > 0 else "Database is empty!"

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=secs, l=l, mess=mess)

@app.route('/deleteunregistereduser', methods=['GET', 'POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form.get('index', -1))
    except ValueError:
        idx = -1

    remove_empty_cells()
    dfu = pd.read_csv('UserList/Unregistered.csv')

    if idx < 0 or idx >= len(dfu):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0,
                               mess="Invalid user index.")

    row = dfu.iloc[idx].copy()
    folder_name = f"{row['Name']}${row['ID']}${row['Section']}"
    folder_path = os.path.join('static/faces', folder_name)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        train_model()
        load_cnn_model()

    dfu.drop(index=idx, inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    remove_empty_cells()
    remAttendance()

    if dfu.empty:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0,
                               mess="Database is empty!")

    names = dfu['Name'].tolist()
    rolls = dfu['ID'].astype(str).tolist()
    secs = dfu['Section'].tolist()
    l = len(dfu)

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=secs, l=l,
                           mess=f'Number of Unregistered Students: {l}')
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username == 'admin' and password == '12345':
            session['admin'] = username
            return redirect(url_for('home'))
        else:
            return render_template('LogInForm.html', mess='Incorrect Username or Password')

    return render_template('LogInForm.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('LogInForm.html')
# Main Function
if __name__ == '__main__':
    # Load or train model
    load_cnn_model()
    
    if cnn_model is None:
        print("[INFO] No model found, will train when needed.")
    app.run(port=5001, debug=True)