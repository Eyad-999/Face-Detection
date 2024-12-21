
# # Import the necessary libraries
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# #from IPython import get_ipython
#
# #get_ipython().magic('matplotlib inline')
#
#
#
# #  Loading the image to be tested
# test_image = cv2.imread(r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\baby1.png')
#
#
# # Converting to grayscale as opencv expects detector takes in input gray scale images
# test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#
# # Displaying grayscale image
# plt.imshow(test_image_gray, cmap='gray')
#
#
# # Since we know that OpenCV loads an image in BGR format so we need to convert it into RBG format to be able to display its true colours. Let us write a small function for that.
#
#
#
# def convertToRGB(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#
# # # Haar cascade files
#
# # Loading the classifier for frontal face
#
#
# haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
#
#
# # # Face detection
#
#
# faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)
#
# # Let us print the no. of faces found
# print('Faces found: ', len(faces_rects))
#
#
#
# # Our next step is to loop over all the co-ordinates it returned and draw rectangles around them using Open CV.We will be drawing a green rectangle with thicknessof 2
#
#
#
# for (x,y,w,h) in faces_rects:
#      cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#
#
# # Finally, we shall display the original image in coloured to see if the face has been detected correctly or not.
#
#
# #convert image to RGB and show image
# plt.imshow(convertToRGB(test_image))
#
#
# #  Let us create a generalised function for the entire face detection process.
#
#
# def detect_faces(cascade, test_image, scaleFactor = 1.1):
#     # create a copy of the image to prevent any changes to the original one.
#     image_copy = test_image.copy()
#
#     #convert the test image to gray scale as opencv face detector expects gray images
#     gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
#
#     # Applying the haar classifier to detect faces
#     faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
#
#     for (x, y, w, h) in faces_rect:
#         cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
#
#     return image_copy
#
#
# # Testing the function on new image
#
#
# #loading image
# test_image2 = cv2.imread(r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\group.png')
#
# #call the function to detect faces
# faces = detect_faces(haar_cascade_face, test_image2)
#
# #convert to RGB and display image
# plt.imshow(convertToRGB(faces))
#
# # Saving the final image
#
# cv2.imwrite('image1.png',faces)
#
#
#########################################################################
# # Import the necessary libraries
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
#
# # Function to load image and check for errors
# def load_image(img_path):
#     if os.path.exists(img_path):
#         image = cv2.imread(img_path)
#         if image is None:
#             raise ValueError("Error: Could not load image. Ensure the file is a valid image format.")
#         return image
#     else:
#         raise FileNotFoundError(f"Error: Path does NOT exist: {img_path}")
#
# # Function to convert BGR image to RGB
# def convertToRGB(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Function to preprocess image (grayscale, histogram equalization)
# def preprocess_image(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply histogram equalization to improve contrast
#     equalized_image = cv2.equalizeHist(gray_image)
#     return equalized_image
#
# # Function to detect faces using a Haar Cascade and count them
# def detect_and_count_faces(cascade, test_image, scaleFactor=1.2, minNeighbors=4):
#     image_copy = test_image.copy()
#     preprocessed_image = preprocess_image(image_copy)
#     faces_rect = cascade.detectMultiScale(preprocessed_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
#     face_count = len(faces_rect)
#
#     for (x, y, w, h) in faces_rect:
#         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Display the number of faces found on the image
#     cv2.putText(image_copy, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     return image_copy, face_count
#
# # Paths to the images and cascade file
# img_path1 = r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\baby1.png'
# img_path2 = r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\group.png'
# cascade_path = r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\haarcascades\haarcascade_frontalface_alt2.xml'
#
# # Load the Haar cascade
# haar_cascade_face = cv2.CascadeClassifier(cascade_path)
#
# # Load and process the first image
# test_image1 = load_image(img_path1)
# test_image_gray = cv2.cvtColor(test_image1, cv2.COLOR_BGR2GRAY)
#
# # Detect and count faces in the first image
# faces_rects1 = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.2, minNeighbors=4)
# print('Faces found in first image: ', len(faces_rects1))
#
# # Draw rectangles around detected faces
# for (x, y, w, h) in faces_rects1:
#     cv2.rectangle(test_image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # Display the first image with detected faces
# plt.imshow(convertToRGB(test_image1))
# plt.title("Detected Faces - Image 1")
# plt.axis('off')
# plt.show()
#
# # Detect and count faces in the second image using the generalized function
# test_image2 = load_image(img_path2)
# faces_detected, face_count2 = detect_and_count_faces(haar_cascade_face, test_image2, scaleFactor=1.2, minNeighbors=4)
#
# # Display the second image with detected faces and count
# plt.imshow(convertToRGB(faces_detected))
# plt.title(f"Detected Faces - Image 2 (Total Faces: {face_count2})")
# plt.axis('off')
# plt.show()
#
# # Save the final image with face count
# cv2.imwrite('image1.png', faces_detected)


############################################################################################
# Import the necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# Function to load image and check for errors
def load_image(img_path):
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Error: Could not load image. Ensure the file is a valid image format.")
        return image
    else:
        raise FileNotFoundError(f"Error: Path does NOT exist: {img_path}")

# Function to convert BGR image to RGB
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to preprocess image (grayscale, histogram equalization)
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to improve contrast
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

# Function to detect faces using a Haar Cascade and count them
def detect_and_count_faces(cascade, test_image, scaleFactor=1.2, minNeighbors=4):
    image_copy = test_image.copy()
    preprocessed_image = preprocess_image(image_copy)
    faces_rect = cascade.detectMultiScale(preprocessed_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    face_count = len(faces_rect)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the number of faces found on the image
    cv2.putText(image_copy, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image_copy, face_count

# Paths to the images and cascade file
img_path1 = r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\baby1.png'
img_path2 = r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\group.png'
cascade_path = r'C:\Users\Eyad.hatem\Desktop\Face-Detection-in-Python-using-OpenCV-master\data\haarcascades\haarcascade_frontalface_alt2.xml'

# Load the Haar cascade
haar_cascade_face = cv2.CascadeClassifier(cascade_path)

# Load and process the first image
test_image1 = load_image(img_path1)
test_image_gray = cv2.cvtColor(test_image1, cv2.COLOR_BGR2GRAY)

# Detect and count faces in the first image
faces_rects1 = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.2, minNeighbors=4)
print('Faces found in first image: ', len(faces_rects1))

# Draw rectangles around detected faces
for (x, y, w, h) in faces_rects1:
    cv2.rectangle(test_image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the first image with detected faces
plt.imshow(convertToRGB(test_image1))
plt.title("Detected Faces - Image 1")
plt.axis('off')
plt.show()

# Detect and count faces in the second image using the generalized function
test_image2 = load_image(img_path2)
faces_detected, face_count2 = detect_and_count_faces(haar_cascade_face, test_image2, scaleFactor=1.2, minNeighbors=4)

# Display the second image with detected faces and count
plt.imshow(convertToRGB(faces_detected))
plt.title(f"Detected Faces - Image 2 (Total Faces: {face_count2})")
plt.axis('off')
plt.show()

# Save the final image with face count
cv2.imwrite('image1.png', faces_detected)

# Define the ground truth face counts (the true number of faces in the images)
true_face_count1 = 1  # Actual number of faces in img_path1
true_face_count2 = 3  # Actual number of faces in img_path2

# Calculate accuracy for each image based on the detected face count
def calculate_accuracy(true_count, predicted_count):
    if true_count == 0 and predicted_count == 0:
        return 1  # Perfect accuracy when no faces are expected and none are detected
    if true_count == 0 or predicted_count == 0:
        return 0  # If there are faces expected but none are detected (or vice versa), accuracy is 0
    return min(true_count, predicted_count) / max(true_count, predicted_count)

# Accuracy for the first image
accuracy1 = calculate_accuracy(true_face_count1, len(faces_rects1))
print(f'Accuracy for Image 1: {accuracy1 * 100:.2f}%')

# Accuracy for the second image
accuracy2 = calculate_accuracy(true_face_count2, face_count2)
print(f'Accuracy for Image 2: {accuracy2 * 100:.2f}%')
