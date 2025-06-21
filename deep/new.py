import cv2
import os

# Function to extract frames from video
def extract_frames(video_path, output_folder, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

# Function to detect and crop faces from images
def crop_faces(input_folder, output_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Example usage
video_path = 'uploads/tomo.mp4'
frames_output_folder = 'frame'
faces_output_folder = 'face folder'

extract_frames(video_path, frames_output_folder)
crop_faces(frames_output_folder, faces_output_folder)