import os
import csv
import cv2
import shutil
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from flask import Flask,render_template,redirect,request,session,g,url_for
from datetime import datetime, date
import joblib
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

load_dotenv()
# Initializing the flask app

app=Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Flask error handler
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('Error.html')


def load_csv_safely(csv_path, expected_columns):
    # If file doesn't exist or empty, create with headers
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"[Fix] File missing or empty: {csv_path}. Creating new with headers.")
        with open(csv_path, 'w') as f:
            f.write(','.join(expected_columns) + '\n')

    try:
        df = pd.read_csv(csv_path)
        # Check if expected columns are all present (order can differ)
        if not set(expected_columns).issubset(df.columns):
            print(f"[Fix] File '{csv_path}' missing expected columns. Resetting.")
            df = pd.DataFrame(columns=expected_columns)
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[Fix] Error reading {csv_path}: {e}. Resetting.")
        df = pd.DataFrame(columns=expected_columns)
        df.to_csv(csv_path, index=False)

    return df

    
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
eye_detector = cv2.CascadeClassifier('static/haarcascade_eye.xml')
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
        
# ======= Remove Empty Rows From Excel Sheet =======
def remove_empty_cells():
    try:
        for file in ['UserList/Registered.csv', 'UserList/Unregistered.csv']:
            if os.path.isfile(file):
                df = pd.read_csv(file)
                df.dropna(how='any', inplace=True)
                df = df[df.apply(lambda row: all(str(cell).strip() != '' for cell in row), axis=1)]
                df.to_csv(file, index=False, header=True)

        if os.path.isdir('Attendance'):
            for file in os.listdir('Attendance'):
                file_path = f'Attendance/{file}'
                try:
                    df = pd.read_csv(file_path)
                    df.dropna(how='any', inplace=True)
                    df = df[df.apply(lambda row: all(str(cell).strip() != '' for cell in row), axis=1)]
                    df.to_csv(file_path,index=False, header=True)
                except pd.errors.EmptyDataError:
                    print(f"[Warning] Empty file skipped: {file_path}")
                except Exception as e:
                    print(f"[Error] Failed processing {file_path}: {e}")

    except Exception as e:
        print(f"[Error] While cleaning CSVs: {e}")
        
# ======= Total Registered Users ========
def totalreg():
    return len(os.listdir('static/faces'))


# ======= Get Face From Image =========
def extract_faces_and_eyes(img):
    if img is None:
        return (), ()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_points = face_detector.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=7)

    eyes_list = []  # List of eyes per face
    for (x, y, w, h) in face_points:
        roi_gray = gray_img[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        # Append eyes relative to the original image
        eyes_global = [(ex+x, ey+y, ew, eh) for (ex, ey, ew, eh) in eyes]
        eyes_list.append(eyes_global)

    return face_points, eyes_list


# Load the CNN model globally once
cnn_model = load_model('final_model/face_recognition_model.h5')

# Get class names (assuming each folder in 'static/faces' is a user/class)
class_names = sorted([
    d for d in os.listdir('static/faces')
    if os.path.isdir(os.path.join('static/faces', d)) and not d.startswith('.')
])

# ======= Identify Face Using ML ========
def identify_face(face_img):
    global cnn_model, class_names
    if face_img is None:
        return "No face detected"
    
    # Resize & preprocess
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    
    # Predict
    preds = cnn_model.predict(face_img)
    pred_index = np.argmax(preds)
    
    # Return class name
    return class_names.get(pred_index, "Unknown")



# ======= Train Model Using Available Faces ========
from tensorflow.keras.models import load_model
import pickle

# Global variables
model = None
class_names = None

def train_model():
    global cnn_model, class_names
    
    face_dir = 'static/faces'
    if not os.path.exists(face_dir) or len(os.listdir(face_dir)) == 0:
        print("No faces to train on.")
        return
    
    # Remove old model if exists
    if os.path.exists('final_model/face_recognition_model.h5'):
        os.remove('final_model/face_recognition_model.h5')
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1
    )
    
    # Generators
    train_data = datagen.flow_from_directory(
        face_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_data = datagen.flow_from_directory(
        face_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Build model
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
    ])
    
    cnn_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"]
    )
    
    # Train
    cnn_model.fit(
        train_data,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        epochs=7
    )
    
    # Save model & labels
    os.makedirs('final_model', exist_ok=True)
    cnn_model.save('final_model/face_recognition_model.h5')
    
    # Save class labels (index -> class name)
    class_names = {v: k for k, v in train_data.class_indices.items()}
    with open('final_model/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    # Reload into memory (ensures immediate recognition)
    cnn_model = load_model('final_model/face_recognition_model.h5')
    with open('final_model/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    
    print("✅ Model trained and reloaded successfully!")



cnn_model = None
class_names = []

def load_cnn_model():
    global cnn_model, class_names

    model_path = 'final_model/face_recognition_model.h5'
    class_names_path = 'final_model/class_names.pkl'

    # Load CNN model
    if os.path.exists(model_path):
        cnn_model = load_model(model_path)
        print(f"[INFO] Loaded CNN model from {model_path}")
    else:
        print(f"[WARNING] CNN model not found at {model_path}.")
        cnn_model = None

    # Load or regenerate class_names.pkl
    if os.path.exists(class_names_path):
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
        print(f"[INFO] Loaded class_names from {class_names_path}")
    else:
        print(f"[WARNING] class_names.pkl not found. Regenerating from Registered.csv...")
        csv_path = 'UserList/Registered.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            class_names = {i: name for i, name in enumerate(df['Name'].tolist())}

            # Ensure final_model directory exists
            os.makedirs(os.path.dirname(class_names_path), exist_ok=True)

            # Save class_names.pkl
            with open(class_names_path, 'wb') as f:
                pickle.dump(class_names, f)
            print(f"[INFO] class_names.pkl created with {len(class_names)} entries.")
        else:
            print(f"[ERROR] Registered.csv not found. Cannot create class_names.pkl.")
            class_names = {}

# ======= Remove Attendance of Deleted User ======
def remAttendance():
    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])
    
    valid_ids = set(map(str, dfu['ID'])) | set(map(str, dfr['ID']))

    for file in os.listdir('Attendance'):
        file_path = f'Attendance/{file}'
        if not file.endswith('.csv'):
            continue

        df = pd.read_csv(file_path)

        # Filter only rows where ID exists in either user list
        df_filtered = df[df['ID'].astype(str).isin(valid_ids)]

        df_filtered.to_csv(file_path, index=False, header=True)

    remove_empty_cells()


# ======== Get Info From Attendance File =========
def extract_attendance():
    datetoday = date.today().strftime("%Y-%m-%d")
    attendance_path = f'Attendance/{datetoday}.csv'

    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    # Check if today's attendance file exists and is not empty
    if not os.path.exists(attendance_path) or os.path.getsize(attendance_path) == 0:
        return [], [], [], [], datetoday, [], 0
    
    df = pd.read_csv(f'Attendance/{datetoday}.csv')

    names = df['Name']
    rolls = df['ID']
    sec = df['Section']
    times = df['Time']
    dates = f'{datetoday}'

    unreg_ids = set(map(str, dfu['ID']))
    reg_ids = set(map(str, dfr['ID']))

    reg = []
    for user_id in map(str, rolls):
        if user_id in unreg_ids:
            reg.append("Unregistered")
        elif user_id in reg_ids:
            reg.append("Registered")
        else:
            reg.append("Unknown")  # optional fallback

    l = len(df)
    return names, rolls, sec, times, dates, reg, l


# ======== Save Attendance =========
def add_attendance(name):
    username, userid, usersection = name.split('$')
    current_time = datetime.now().strftime("%I:%M %p")
    
    file_path = f'Attendance/{datetoday}.csv'
    
    # If file doesn't exist or is empty, create with headers
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        df = pd.DataFrame(columns=['Name', 'ID', 'Section', 'Time'])
        df.to_csv(file_path, index=False)
    
    # Now safely read the file
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        # In case file became empty meanwhile, recreate and read again
        df = pd.DataFrame(columns=['Name', 'ID', 'Section', 'Time'])
        df.to_csv(file_path, index=False)
        df = pd.read_csv(file_path)
    
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
        except Exception as e:
            print(f"Time parsing error, adding attendance anyway: {e}")
            new_entry = pd.DataFrame([[username, userid, usersection, current_time]],
                                     columns=['Name', 'ID', 'Section', 'Time'])
            new_entry.to_csv(file_path, mode='a', index=False, header=False)



# ======= Flask Home Page =========
@app.route('/')
def home():
    if g.user:
        return render_template('HomePage.html', admin=True, mess='Logged in as Administrator', user=session['admin'])

    return render_template('HomePage.html', admin=False, datetoday2=datetoday2)

@app.route('/attendance')
def take_attendance():
    if f'{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/{datetoday}.csv', 'w') as f:
            f.write('Name,ID,Section,Time')

    remove_empty_cells()
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)
    
    
@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    if len(os.listdir('static/faces')) == 0:
        return render_template('Attendance.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

    if 'face_recognition_model.h5' not in os.listdir('final_model'):
        train_model()

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    ret = True
    j = 1
    flag = None

    while ret:
        ret, frame = cap.read()
        faces, eyes_list = extract_faces_and_eyes(frame)  # Detect faces and eyes

        identified_person_name = "Unknown"
        identified_person_id = "N/A"

        if faces is not None and len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                # Only consider faces that have at least one detected eye
                if i < len(eyes_list) and len(eyes_list[i]) > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)

                    # Draw rectangles around eyes
                    for (ex, ey, ew, eh) in eyes_list[i]:
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    face_img = cv2.resize(frame[y:y + h, x:x + w], (224, 224))
                    identified_person = identify_face(face_img)

                    if identified_person is not None and '$' in identified_person:
                        identified_person_name, identified_person_id, *_ = identified_person.split('$')

                        if flag != identified_person:
                            j = 1
                            flag = identified_person

                        if j % 20 == 0:
                            add_attendance(identified_person)
                    else:
                        identified_person_name = "Unknown"
                        identified_person_id = "N/A"

                    cv2.putText(frame, f'Name: {identified_person_name}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'ID: {identified_person_id}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'Press Esc to close', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 127, 255), 2, cv2.LINE_AA)

                    j += 1
        else:
            j = 1
            flag = None

        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance', 800, 600)
        cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:  # ESC key
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
    
    # CSV paths
    csv_unreg_path = os.path.join(app.root_path, 'UserList', 'Unregistered.csv')
    csv_reg_path = os.path.join(app.root_path, 'UserList', 'Registered.csv')
    
    dfu = load_csv_safely(csv_unreg_path, ['Name', 'ID', 'Section'])
    dfr = load_csv_safely(csv_reg_path, ['Name', 'ID', 'Section'])
    
    # Check duplicates
    if str(newuserid) in map(str, dfu['ID']) or str(newuserid) in map(str, dfr['ID']):
        cap.release()
        return render_template('AddUser.html', mess='User already exists.')
    
    # Capture face images
    images_captured = 0
    frame_count = 0
    max_frames = 1000
    while images_captured < 50 and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces, eyes_list = extract_faces_and_eyes(frame)
        if faces is not None:
            for i, (x, y, w, h) in enumerate(faces):
                if i < len(eyes_list) and len(eyes_list[i]) > 0:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (224,224))
                    if frame_count % 10 == 0:
                        cv2.imwrite(os.path.join(userimagefolder, f'{newusername}_{images_captured}.jpg'), face_img)
                        images_captured += 1
                    # Draw rectangles
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,20), 2)
                    for (ex, ey, ew, eh) in eyes_list[i]:
                        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
                    cv2.putText(frame, f'Images Captured: {images_captured}/50', (30,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,20), 2)
        
        cv2.imshow('Collecting Face Data', frame)
        if cv2.waitKey(1) == 27:
            break
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    if images_captured == 0:
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture valid face images.')
    
    # Append to Unregistered.csv
    with open(csv_unreg_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([newusername, newuserid, newusersection])
    
    # Retrain model immediately with new user included
    train_model()
    
    # Reload unregistered users list
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
    return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg,
                           l=l)
    
# ========== Flask Search Attendance by Date ============
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('LogInForm.html')

    date = request.form['date']

    year = date.split('-')[0]
    month = date.split('-')[1]
    day = date.split('-')[2]

    if f'{day}-{month}-{year}.csv' not in os.listdir('Attendance'):
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=0,
                               mess="Nothing Found!")
    else:
        names = []
        rolls = []
        sec = []
        times = []
        dates = []
        reg = []
        l = 0

        skip_header = True
        csv_file = csv.reader(open(f'Attendance/{day}-{month}-{year}.csv', "r"), delimiter=",")
        dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
        dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

        for row in csv_file:
            if skip_header:
                skip_header = False
                continue

            names.append(row[0])
            rolls.append(row[1])
            sec.append(row[2])
            times.append(row[3])
            dates.append(f'{day}-{month}-{year}')

            if str(row[1]) in list(map(str, dfu['ID'])):
                reg.append("Unregistered")
            elif str(row[1]) in list(map(str, dfr['ID'])):
                reg.append("Registered")
            else:
                reg.append("x")

            l += 1

        if l != 0:
            return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                                   reg=reg, l=l,
                                   totalreg=totalreg())
        else:
            return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                                   reg=reg, l=l,
                                   totalreg=totalreg(),
                                   mess="Nothing Found!")

# ========== Flask Search Attendance by ID ============
@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('LogInForm.html')

    id = request.form.get('id')
    if not id:
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
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            try:
                with open(filepath, "r") as f:
                    csv_file = csv.reader(f, delimiter=",")
                    for row in csv_file:
                        if len(row) > 3 and row[1] == id:  # Check row length to avoid index error
                            names.append(row[0])
                            rolls.append(row[1])
                            sec.append(row[2])
                            times.append(row[3])
                            dates.append(filename.replace('.csv', ''))

                            if str(row[1]) in list(map(str, dfu['ID'])):
                                reg.append("Unregistered")
                            elif str(row[1]) in list(map(str, dfr['ID'])):
                                reg.append("Registered")
                            else:
                                reg.append("Unknown")
                            count += 1
            except Exception as e:
                print(f"[Warning] Skipping file {filename} due to error: {e}")

    if count > 0:
        return render_template('AttendanceList.html',
                               names=names, rolls=rolls, sec=sec,
                               times=times, dates=dates, reg=reg,
                               l=count,
                               mess=f'Total Attendance: {count}')
    else:
        return render_template('AttendanceList.html',
                               names=[], rolls=[], sec=[],
                               times=[], dates=[], reg=[],
                               l=0,
                               mess="Nothing Found!")


@app.route('/registeruserlist')
def register_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        if len(row) != 3:
            print(f"[Warning] Skipping malformed row: {row}")
            continue

        names.append(row[0].strip())
        rolls.append(row[1].strip())
        sec.append(row[2].strip())
        l += 1

    if l != 0:
        return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")

# ========== Flask Unregister a User ============
@app.route('/unregisteruser', methods=['POST'])
def unregisteruser():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()
    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index (not a number or missing)", 400

    remove_empty_cells()
    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])
    

    if idx < 0 or idx >= len(dfr):
        return f"Invalid index: {idx} for {len(dfr)} users", 400
    
    # Extract the row to unregister
    row = dfr.iloc[idx].copy()

    # Move the user's face folder, handle if exists or not
    old_folder = f"static/faces/{row['Name']}${row['ID']}${row['Section']}"
    new_folder = f"static/faces/{row['Name']}${row['ID']}$None"

    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)  # Remove old None folder if exists to avoid conflicts
        shutil.move(old_folder, new_folder)

    # Change Section value to 'None'
    row['Section'] = 'None'

    # Add to unregistered DataFrame
    dfu = pd.concat([dfu, pd.DataFrame([row])], ignore_index=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False, header=True)

    # Remove from registered DataFrame
    dfr = dfr.drop(dfr.index[idx])
    dfr.to_csv('UserList/Registered.csv', index=False, header=True)

    train_model()
    remove_empty_cells()

    # Use updated DataFrame to get user info to pass to template
    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    names = dfr['Name'].tolist()
    rolls = dfr['ID'].astype(str).tolist()
    sec = dfr['Section'].tolist()
    l = len(dfr)

    mess = f'Number of Registered Students: {l}' if l > 0 else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)

# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['GET', 'POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    remove_empty_cells()
    dfr = pd.read_csv('UserList/Registered.csv')
    username = dfr.iloc[idx]['Name']
    userid = dfr.iloc[idx]['ID']
    usersec = dfr.iloc[idx]['Section']

    if f'{username}${userid}${usersec}' in os.listdir('static/faces'):
        shutil.rmtree(f'static/faces/{username}${userid}${usersec}')
        train_model()

    dfr.drop(dfr.index[idx], inplace=True)
    dfr.to_csv('UserList/Registered.csv', index=False, header=True)

    remove_empty_cells()

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    remAttendance()

    if l != 0:
        return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")
        
@app.route('/unregisteruserlist')
def unregister_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    remove_empty_cells()

    df = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    if df.empty:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0,
                               mess="Database is empty!")

    # Extract columns as lists
    names = df['Name'].tolist()
    rolls = df['ID'].astype(str).tolist()  # Convert to string if needed
    sec = df['Section'].tolist()
    l = len(df)
    
    return render_template('UnregisterUserList.html',names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

# ========== Flask Register a User ============
@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])
    new_section = request.form['section']

    remove_empty_cells()

    # Read CSVs into dataframes
    dfr = load_csv_safely('UserList/Registered.csv', ['Name', 'ID', 'Section'])
    dfu = load_csv_safely('UserList/Unregistered.csv', ['Name', 'ID', 'Section'])

    if idx >= len(dfu):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    # Extract user details
    row = dfu.iloc[idx]
    name, uid, _ = row['Name'], row['ID'], row['Section']

    old_folder = f'static/faces/{name}${uid}$None'
    new_folder = f'static/faces/{name}${uid}${new_section}'

    # Move face data folder
    if os.path.exists(old_folder):
        shutil.move(old_folder, new_folder)

    # Update section and append to registered
    row['Section'] = new_section
    dfr = pd.concat([dfr, pd.DataFrame([row])], ignore_index=True)
    dfr.to_csv('UserList/Registered.csv', index=False, header=True)

    # Remove from unregistered
    dfu = dfu.drop(index=idx)
    dfu.to_csv('UserList/Unregistered.csv', index=False, header=True)

    # Retrain model
    train_model()
    load_cnn_model()
    remove_empty_cells()

    # Render updated unregistered list
    if dfu.empty:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Database is empty!")

    names = dfu['Name'].tolist()
    rolls = dfu['ID'].astype(str).tolist()
    secs = dfu['Section'].tolist()
    l = len(dfu)

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=secs, l=l,
                           mess=f'Number of Unregistered Students: {l}')



# ========== Flask Delete a User from Unregistered List ============
@app.route('/deleteunregistereduser', methods=['GET', 'POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
    except (KeyError, ValueError):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid index.")

    remove_empty_cells()
    dfu = pd.read_csv('UserList/Unregistered.csv')

    # Ensure index is within bounds
    if idx < 0 or idx >= len(dfu):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    # Extract info
    username = dfu.iloc[idx]['Name']
    userid = dfu.iloc[idx]['ID']
    usersec = dfu.iloc[idx]['Section']

    folder_name = f'{username}${userid}${usersec}'
    folder_path = os.path.join('static/faces', folder_name)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        train_model()

    # Drop the row from Unregistered.csv
    dfu.drop(index=idx, inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    remove_empty_cells()
    remAttendance()

    # Re-prepare updated unregistered list
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
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '12345':
            session['admin'] = request.form['username']
            return redirect(url_for('home', admin=True, mess='Logged in as Administrator'))
        else:
            return render_template('LogInForm.html', mess='Incorrect Username or Password')
    return render_template('LogInForm.html')

# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('LogInFrom.html')

# Main Function
if __name__ == '__main__':
    cnn_model = None
    class_names = []
    load_cnn_model()
    app.run(port=5001, debug=True)