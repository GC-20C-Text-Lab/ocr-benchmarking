from PIL import Image

import pytesseract

import os

# This path name is Windows-specific. See if there is a way to include this into the command line's
# PATH or create a conditional to check if the code is running on Windows.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# replace with the name of the folder where the tif files are
directory = "scans-tif"
filenames = []

for filename in os.listdir(directory) :
  if os.path.isfile(os.path.join(directory, filename)) :
    filenames.append(os.path.join(directory, filename))

# This is the directory where the output will be written to. 
# Have a file already created for each page you are OCRing in order for this to work
directory2 = "raw-ocr"
filenames2 = []

for filename in os.listdir(directory2) :
  if os.path.isfile(os.path.join(directory2, filename)) :
    filenames2.append(os.path.join(directory2, filename))

# This for loop matches the image files in the first directory with the text files in the second directory
# As long as all necessary files have been created already and are in the correct order 
# Scans the files with the automatically detected layout (two columns)
# Writes the output to the files in the second directory
for i in range(len(filenames)) :
  with open (filenames2[i], "w") as file:
    file.write(pytesseract.image_to_string(Image.open(filenames[i]), config='--psm 1'))
    
