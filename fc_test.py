from flask import Flask, render_template, request, send_file, after_this_request, redirect, url_for
from threading import Thread
import cv2
import face_recognition
import os
import datetime
import csv
import zipfile
import glob
from werkzeug.utils import secure_filename
import base64
import threading
import time

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start_face_recognition')
def start_face_recognition():
    thread = threading.Thread(target=face_recognition_code)
    thread.start()
    return redirect(url_for('home'))

def face_recognition_code():
    # Load the known images and get their face encodings
    known_face_encodings = []
    known_face_names = []
    images_dir = "images"

    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(filename.rsplit('_', 1)[0])  # Remove the extension from filename

    # Start the webcam feed
    video_capture = cv2.VideoCapture(0)

    # Initialize a set to store the usernames that have been written to the file
    written_usernames = set()

    # Define the font here
    font = cv2.FONT_HERSHEY_DUPLEX

    last_face_detected_time = time.time()  # Initialize the timer

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if face_locations:  # If any face is detected
            last_face_detected_time = time.time()  # Reset the timer

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"
            userid = "Unknown"
            color = (0, 0, 255)  # Red color for unknown faces

            if True in matches:
                first_match_index = matches.index(True)
                name_userid = known_face_names[first_match_index]
                name, userid = name_userid.split('_')  # Extract username and userid from filename

                # Write the details to a CSV file only if the username hasn't been written yet
                filename = now.strftime("%Y-%m-%d") + ".csv"
                found_in_csv = False
                if os.path.exists(filename):
                    with open(filename, 'r') as file:
                        reader = csv.reader(file)
                        if any(userid == row[0] for row in reader):
                            color = (0, 255, 0)  # Green color for known faces
                            found_in_csv = True

                if not found_in_csv:
                    with open(filename, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([userid,name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    written_usernames.add(name)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the current date and time on the video
        cv2.putText(frame, current_time, (10, 30), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        if time.time() - last_face_detected_time > 5 * 60:  # 10 minutes * 60 seconds
            break

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()




# Flask route

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        userid = request.form['userid']
        images = request.files.getlist('images')

        if not os.path.exists('images'):
            os.makedirs('images')

        for i, image in enumerate(images):
            filename = secure_filename(f'{username}_{userid}_{i}.jpg')
            image.save(os.path.join('images', filename))

        return 'Registration successful'
    return render_template('register.html')




@app.route('/download', methods=['POST'])
def download():
    # Get the month and day from the form data
    month = request.form.get('month')
    day = request.form.get('day')

    # Check if day is provided
    if day:
        # Format the filename
        filename = f"{datetime.datetime.now().year}-{int(month):02d}-{int(day):02d}.csv"
    else:
        # If day is not provided, use a wildcard to match any day of the month
        filename = f"{datetime.datetime.now().year}-{int(month):02d}-*.csv"

    # Check if any file matching the pattern exists
    files = glob.glob(filename)
    if not files:
        return "File not found", 404

    # If day is provided, send the single file for download
    if day:
        return send_file(files[0], as_attachment=True)

    # If day is not provided, zip all files of the month and send for download
    zipf = zipfile.ZipFile('Files.zip', 'w', zipfile.ZIP_DEFLATED)
    for file in files:
        zipf.write(file)
    zipf.close()

    @after_this_request
    def remove_file(response):
        try:
            os.remove('Files.zip')
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    return send_file('Files.zip', as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)