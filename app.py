import os
import sqlite3
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Upload Folder Configuration
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure Upload Directory Exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to Check Allowed File Extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize MediaPipe Pose and FaceMesh once to optimize performance
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Database Setup
def create_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                           (name, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Email already registered. Try again."

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):  # Check if password matches the hashed password
            session['user'] = user[1]  # Store name in session
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials. Try again."

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/find-style', methods=['GET', 'POST'])
def find_style():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request."
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file."

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process the image for face detection
            image = cv2.imread(filepath)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect landmarks using MediaPipe Pose and FaceMesh
            pose_results = pose.process(rgb_image)
            face_results = face_mesh.process(rgb_image)

            # Initialize body shape, face shape, and skin tone
            body_shape = "Unknown"
            face_shape = "Unknown"
            skin_tone = "Unknown"

            # Face Shape Detection
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    face_points = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]

                    face_width = np.linalg.norm(np.array(face_points[10]) - np.array(face_points[230]))  # Left cheek to right cheek
                    face_height = np.linalg.norm(np.array(face_points[151]) - np.array(face_points[10]))  # Chin to forehead

                    if face_width / face_height > 1.2:
                        face_shape = "Round"
                    elif face_width / face_height < 1.1:
                        face_shape = "Oval"
                    else:
                        face_shape = "Square"

            # Skin Tone Detection
            face_region = image[50:200, 50:200]  # Use a larger region for better skin detection
            hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)  # Convert to HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Light skin
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)  # Dark skin
            skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
            skin_pixels = cv2.bitwise_and(face_region, face_region, mask=skin_mask)

            if np.sum(skin_mask) > 0:
                avg_hue = np.mean(skin_pixels[:, :, 0])  # Get average Hue channel for detected skin
                if avg_hue < 15:  # Light skin tone
                    skin_tone = "Light"
                elif 15 <= avg_hue <= 30:  # Medium skin tone
                    skin_tone = "Medium"
                else:  # Dark skin tone
                    skin_tone = "Dark"

            # Body Shape Classification using Body Landmarks
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

                shoulder_distance = np.linalg.norm(np.array([shoulder_left.x, shoulder_left.y]) - np.array([shoulder_right.x, shoulder_right.y]))
                hip_distance = np.linalg.norm(np.array([hip_left.x, hip_left.y]) - np.array([hip_right.x, hip_right.y]))
                knee_distance = np.linalg.norm(np.array([knee_left.x, knee_left.y]) - np.array([knee_right.x, knee_right.y]))

                if shoulder_distance > hip_distance and hip_distance > knee_distance:
                    body_shape = "Hourglass"
                elif hip_distance > shoulder_distance and hip_distance > knee_distance:
                    body_shape = "Pear"
                elif shoulder_distance > hip_distance and hip_distance < knee_distance:
                    body_shape = "Apple"
                else:
                    body_shape = "Rectangle"

            # Save the image with landmarks drawn
            result_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
            annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_image_path, annotated_image)

            # Return result
            return render_template("result.html", image_url=result_image_path, skin_tone=skin_tone, face_shape=face_shape, body_shape=body_shape)

    return render_template('find_style.html')

@app.route('/select_occasion', methods=['GET', 'POST'])
def select_occasion():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        occasion = request.form['occasion']  # Get the selected occasion (e.g., Wedding, Party, etc.)
        file = request.files['file']
        
        if file.filename == '':
            return "No file selected."
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process the image for face and body detection (same as "find-style")
            image = cv2.imread(filepath)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect body landmarks
            pose_results = pose.process(rgb_image)
            face_results = face_mesh.process(rgb_image)

            # Initialize skin tone, face shape, and body shape
            skin_tone = "Unknown"
            face_shape = "Unknown"
            body_shape = "Unknown"

            # Face Shape Detection using MediaPipe FaceMesh
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    face_points = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]
                    face_width = np.linalg.norm(np.array(face_points[10]) - np.array(face_points[230]))
                    face_height = np.linalg.norm(np.array(face_points[151]) - np.array(face_points[10]))

                    if face_width / face_height > 1.2:
                        face_shape = "Round"
                    elif face_width / face_height < 1.1:
                        face_shape = "Oval"
                    else:
                        face_shape = "Square"

            # Skin Tone Detection
            face_region = image[50:200, 50:200]
            hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
            skin_pixels = cv2.bitwise_and(face_region, face_region, mask=skin_mask)

            if np.sum(skin_mask) > 0:
                avg_hue = np.mean(skin_pixels[:, :, 0])
                if avg_hue < 15:
                    skin_tone = "Light"
                elif 15 <= avg_hue <= 30:
                    skin_tone = "Medium"
                else:
                    skin_tone = "Dark"

            # Body Shape Classification using Body Landmarks
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

                shoulder_distance = np.linalg.norm(np.array([shoulder_left.x, shoulder_left.y]) - np.array([shoulder_right.x, shoulder_right.y]))
                hip_distance = np.linalg.norm(np.array([hip_left.x, hip_left.y]) - np.array([hip_right.x, hip_right.y]))
                knee_distance = np.linalg.norm(np.array([knee_left.x, knee_left.y]) - np.array([knee_right.x, knee_right.y]))

                if shoulder_distance > hip_distance and hip_distance > knee_distance:
                    body_shape = "Hourglass"
                elif hip_distance > shoulder_distance and hip_distance > knee_distance:
                    body_shape = "Pear"
                elif shoulder_distance > hip_distance and hip_distance < knee_distance:
                    body_shape = "Apple"
                else:
                    body_shape = "Rectangle"

            result_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
            annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_image_path, annotated_image)

            return render_template("result2.html", image_url=result_image_path, skin_tone=skin_tone, face_shape=face_shape, body_shape=body_shape, occasion=occasion)

    return render_template('select_occasion.html')

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    create_database()  # Run database creation
    app.run(debug=True)
