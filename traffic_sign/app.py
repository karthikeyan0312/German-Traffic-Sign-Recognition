import streamlit as st
from tensorflow.keras import models
import cv2 as cv
import numpy as np
from PIL import Image
import validators,requests,urllib
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load():
    path=r"model/gtsrb_model"
    return models.load_model(path)
    

def is_url_image(image_url):
   image_formats = ("image/png", "image/jpeg", "image/jpg")
   r = requests.head(image_url)
   if r.headers["content-type"] in image_formats:
      return True
   return False

model=load()
st.title("Traffic sign Detection")
st.markdown("****")

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

selectBox = st.sidebar.selectbox(
    'How would you like to check sign?',
    ('Upload File', 'Enter  URL')
)

if selectBox=='Upload File':
    upload=st.file_uploader('Choose a File ',type=["jpg","jpeg","png"])
    if upload  is not None:
        try:
            file_bytes = np.asarray(bytearray(upload.read()),dtype=np.uint8)
            image = cv.imdecode(file_bytes, 1)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            img_array=np.array(resize_image)/255
            img_array=img_array.reshape(1, 30, 30, -1)
            p=np.argmax(model.predict(img_array))
            st.success("Detected Sign Board is : "+classes[p])
        except Exception as e:
            st.warning("You Have Upladed Wrong Image")
    else:
        st.info("Upload Image")
elif selectBox=='Enter  URL':
    url=st.text_input('Enter some text')
    #st.write(url)
    if validators.url(url):
        if is_url_image(url):
            try:
                req = urllib.request.urlopen(url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                img = cv.imdecode(arr, -1) # 'Load it as it is'
                #image_fromarray = Image.fromarray(img, 'RGB')
                resize_image = Image.fromarray(img, 'RGB').resize((IMG_HEIGHT, IMG_WIDTH))
                reshaped=np.array(resize_image)/255
                p=np.argmax(model.predict(reshaped.reshape(1,30,30,-1)))
                st.success("Detected Sign Board is : "+classes[p])
            except:
                st.warning("Something Went Wrong, try again")
        else:
            st.warning("This URL does not have any Images")
    elif not url:
        st.info("Enter URL")
    else:
        st.warning("Enter a valid url")

