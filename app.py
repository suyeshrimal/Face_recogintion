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
        
# ======= Remove Empty Rows From Excel Sheet =======
def remove_empty_cells():
    try:
        for file in ['UserList/Registered.csv', 'UserList/Unregistered.csv']:
            if os.path.isfile(file):
                df = pd.read_csv(file)
                df.dropna(inplace=True)
                df.to_csv(file, index=False)

        if os.path.isdir('Attendance'):
            for file in os.listdir('Attendance'):
                file_path = f'Attendance/{file}'
                try:
                    df = pd.read_csv(file_path)
                    df.dropna(inplace=True)
                    df.to_csv(file_path, index=False)
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
def extract_faces(img):
    if img is None:
        return ()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=7)
    return face_points


# Load the CNN model globally once
cnn_model = load_model('static/face_recognition_model.h5')

# Get class names (assuming each folder in 'static/faces' is a user/class)
class_names = sorted([
    d for d in os.listdir('static/faces')
    if os.path.isdir(os.path.join('static/faces', d)) and not d.startswith('.')
])

# ======= Identify Face Using ML ========
def identify_face(face_img):
    # face_img is expected to be a BGR image (from OpenCV)
    if face_img is None:
        return "No face detected"
    # Resize to (224, 224) because your CNN expects this size
    face_img = cv2.resize(face_img, (224, 224))
    
    # Convert BGR to RGB because Keras models usually expect RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixels to [0,1]
    face_img = face_img.astype('float32') / 255.0
    
    # Expand dims to shape (1, 224, 224, 3)
    face_img = np.expand_dims(face_img, axis=0)
    
    # Predict class probabilities
    preds = cnn_model.predict(face_img)
    
    # Get the index of highest probability
    pred_index = np.argmax(preds)
    
     # Debug info
    print("Predictions:", preds)
    print("Predicted index:", pred_index)
    print("Class names:", class_names)
    
    # Get the class name
    # predicted_class = class_names[pred_index]
    
    # return predicted_class
    if 0 <= pred_index < len(class_names):
        return class_names[pred_index]
    else:
        return "Unknown"

# ======= Train Model Using Available Faces ========
def train_model():
    face_dir = 'static/faces'
    if len(os.listdir(face_dir)) == 0:
        return

    # Remove old CNN model if exists
    if 'face_recognition_model.h5' in os.listdir('static'):
        os.remove('static/face_recognition_model.h5')

    # Data generator for training
    datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=0.2  # Split training and validation
    )

    train_data = datagen.flow_from_directory(
        face_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        face_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Simple CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(10, kernel_size=3, activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(10, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(10, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        epochs=5,
        validation_data=val_data
    )

    # Save the model
    model.save('static/face_recognition_model.h5')

# ======= Remove Attendance of Deleted User ======
def remAttendance():
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    valid_ids = set(map(str, dfu['ID'])) | set(map(str, dfr['ID']))

    for file in os.listdir('Attendance'):
        file_path = f'Attendance/{file}'
        if not file.endswith('.csv'):
            continue

        df = pd.read_csv(file_path)

        # Filter only rows where ID exists in either user list
        df_filtered = df[df['ID'].astype(str).isin(valid_ids)]

        df_filtered.to_csv(file_path, index=False)

    remove_empty_cells()


# ======== Get Info From Attendance File =========
def extract_attendance():
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')
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

    remove_empty_cells()
    file_path = f'Attendance/{datetoday}.csv'
    
    # Check if file exists, else create with headers
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['Name', 'ID', 'Section', 'Time'])
        df.to_csv(file_path, index=False)
        
    df = pd.read_csv(file_path)

    # Check if user has any attendance today
    user_rows = df[df['ID'].astype(str) == str(userid)]

    if user_rows.empty:
        new_entry = pd.DataFrame([[username, userid, usersection, current_time]],
                                 columns=['Name', 'ID', 'Section', 'Time'])
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # Check time difference
        last_time = user_rows.iloc[-1]['Time']
        start_time = datetime.strptime(last_time, "%I:%M %p")
        end_time = datetime.strptime(current_time, "%I:%M %p")
        delta = (end_time - start_time).total_seconds() / 60

        if delta > 40:
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

    if 'face_recognition_model.h5' not in os.listdir('static'):
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
        faces = extract_faces(frame)  # Detect all faces

        identified_person_name = "Unknown"
        identified_person_id = "N/A"

        if faces is not None and len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use first detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)

            # Resize and prepare the face
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
                identified_person = "Unknown"

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

        # Display the frame
        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance', 800, 600)
        cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:  # ESC key to exit
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
    # newusersection = request.form['newusersection']

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return render_template('AddUser.html', mess='Camera not available.')

    userimagefolder = f'static/faces/{newusername}${newuserid}$None'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    remove_empty_cells()
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    if str(newuserid) in list(map(str, dfu['ID'])):
        cap.release()
        return render_template('AddUser.html', mess='You are already in unregistered list.')
    if str(newuserid) in list(map(str, dfr['ID'])):
        cap.release()
        return render_template('AddUser.html', mess='You are already a registered user.')

    with open('UserList/Unregistered.csv', 'a') as f:
        f.write(f'\n{newusername},{newuserid},None')

    images_captured = 0
    frame_count = 0
    max_frames = 1000  # To avoid infinite loop if faces not detected enough
    skip_count = 0

    while images_captured < 50 and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if faces is None or len(faces) == 0:  # No faces detected in this frame
            skip_count += 1
            # Optional: show message on frame or just continue
            cv2.putText(frame, f'No face detected. Please face the camera.', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            skip_count = 0
            for (x, y, w, h) in faces:
                cv2.putText(frame, f'Images Captured: {images_captured}/50', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                # Save image every 10 frames to avoid similar images
                if frame_count % 10 == 0:
                    img_name = f'{newusername}_{images_captured}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, img_name), frame[y:y + h, x:x + w])
                    images_captured += 1

        cv2.namedWindow('Collecting Face Data', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Collecting Face Data', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Collecting Face Data', frame)

        if cv2.waitKey(1) == 27:  # ESC key to cancel
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # If no images captured, remove user from unregistered list and delete folder
    if images_captured == 0:
        dfu = pd.read_csv('UserList/Unregistered.csv')
        dfu = dfu[dfu['ID'] != newuserid]  # Remove user row
        dfu.to_csv('UserList/Unregistered.csv', index=False)

        remove_empty_cells()

        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture photos.')

    # Train model with new images
    train_model()
     # ✅ Re-read the updated list and pass it to the template
    dfu = pd.read_csv('UserList/Unregistered.csv')
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
        dfu = pd.read_csv('UserList/Unregistered.csv')
        dfr = pd.read_csv('UserList/Registered.csv')

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

    id = request.form['id']

    names = []
    rolls = []
    sec = []
    times = []
    dates = []
    reg = []
    l = 0

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    for file in os.listdir('Attendance'):
        csv_file = csv.reader(open('Attendance/' + file, "r"), delimiter=",")

        for row in csv_file:
            if row[1] == id:
                names.append(row[0])
                rolls.append(row[1])
                sec.append(row[2])
                times.append(row[3])
                dates.append(file.replace('.csv', ''))

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
                               mess=f'Total Attendance: {l}')
    else:
        return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=l,
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

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
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

    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index", 400

    remove_empty_cells()
    dfr = pd.read_csv('UserList/Registered.csv')
    dfu = pd.read_csv('UserList/Unregistered.csv')

    if idx < 0 or idx >= len(dfr):
        return "Index out of range", 400

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
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    # Remove from registered DataFrame
    dfr = dfr.drop(dfr.index[idx])
    dfr.to_csv('UserList/Registered.csv', index=False)

    train_model()
    remove_empty_cells()

    # Use updated DataFrame to get user info to pass to template
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
    dfr.to_csv('UserList/Registered.csv', index=False)

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

    try:
        df = pd.read_csv('UserList/Unregistered.csv')
    except FileNotFoundError:
        # If file doesn't exist or is empty
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0,
                               mess="Database is empty!")

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
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

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
    dfr.to_csv('UserList/Registered.csv', index=False)

    # Remove from unregistered
    dfu = dfu.drop(index=idx)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    # Retrain model
    train_model()
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