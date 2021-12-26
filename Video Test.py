import sys
sys.path.insert(0, '/home/yusqi/Bismillah Skripsi/Video Testing/helper')
import cv2
import numpy as np
from CreateData import create_data
from Prediction import yolo_predict_image
from ShowImage import show_image


# Create a VideoCapture object
cap = cv2.VideoCapture("outpy.avi")

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('save.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True: 
    
    # Write the frame into the file 'output.avi'
    img = create_data(None, frame)
    box = yolo_predict_image(img)
    image = frame.reshape(-1, frame.shape[0], frame.shape[1], frame.shape[2])
    image = show_image(box, image, None, None)

    out.write(image)

    # Display the resulting frame    
    cv2.imshow('frame',image)

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
