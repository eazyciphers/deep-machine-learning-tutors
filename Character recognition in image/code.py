import cv2
import numpy as np
import pytesseract
from PIL import Image

# path of pytesseract execution folder 
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Path of image
main_path = r'C:\Users\dell\Desktop\Image to text\qu12.png'

def get_string(pic_path):
    # Reading picture with opencv
    pic = cv2.imread(pic_path)

    # grey-scale the picture
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    # Do dilation and erosion to eliminate unwanted noises
    kernel = np.ones((1, 1), np.uint8)
    pic = cv2.dilate(pic, kernel, iterations=20)
    pic = cv2.erode(pic, kernel, iterations=20)

    # Write image after removed noise
    cv2.imwrite(main_path + "no_noise.png", pic)

    #  threshold applying to get only black and white picture 
    pic = cv2.adaptiveThreshold(pic, 300, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image for later recognition process 
    cv2.imwrite(main_path + "threshold.png", pic)

    # Character recognition with tesseract
    final = pytesseract.image_to_string(Image.open(main_path + "threshold.png"))

    return final


#starts recognition of characters
print(get_string(src_path))
#displays the output when it recognizes 
