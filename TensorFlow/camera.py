
# import the opencv library
import cv2
from test import predict_animal
from PIL import Image
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
i = 0
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if i%10 == 0:
        # print(frame.shape)
        # print(type(frame))
        frame = cv2.resize(frame, (50, 50))
        predict_animal(frame)
    i += 1

      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()