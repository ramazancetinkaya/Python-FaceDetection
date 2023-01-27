import os
import cv2

def detect_faces(image_path, output_path=None):
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

    # Read the image
    try:
        img = cv2.imread(image_path)
    except Exception as e:
        print(f"Error while reading the image {image_path}: {e}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle and label around the faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"Face {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save the output image with detected faces
    if output_path:
        try:
            cv2.imwrite(output_path, img)
            print(f"Output image saved to {output_path}")
        except Exception as e:
            print(f"Error while saving the output image: {e}")

    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

def process_folder(folder_path, output_folder=None):
    for image_file in os.listdir(folder_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(folder_path, image_file)
            if output_folder:
                output_path = os.path.join(output_folder, image_file)
            else:
                output_path = None
            detect_faces(image_path, output_path)

# Call the function
detect_faces('path/to/image.jpg', 'path/to/output_image.jpg')
# or process a whole folder
process_folder('path/to/folder/')
