import pickle
import streamlit as st
# from streamlit_option_menu import option_menu
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import torchvision
from PIL import Image
import torch
from torchvision import transforms
import time

if "photo" not in st.session_state:
    st.session_state["photo"] ="not done yet"

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True


def change_photo_state():
    st.session_state["photo"] ="done"
    st.session_state.clicked = False

def progress_bar(text):
    progress_text = text
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.5)
    my_bar.empty()

# loading the saved pretrained model
# Pneumonia_model = load_model('Pre-trained_models/cnn_pneumonia_h5_v02.h5')

# Pneumonia_model = pickle.load(open('Pre-trained_models/pneumonia-disease-pretrained-mode.sav','rb'))

# eye_disease_model = pickle.load(open('Pre-trained_models/eye-disease-pretrained-mode.sav','rb'))

# image_classification_model = pickle.load(open('Pre-trained_models/image_classifying_preTrained_model.sav','rb'))

# cnn_pneumonia_model = load_model('Pre-trained_models/cnn_pneumonia_h5_v02.h5')

# # cnn_eye_disease_model = pickle.load(open('Pre-trained_models/CNN-Eye_disease.sav','rb'))

# naive_bayes_pneumonia_model = pickle.load(open('Pre-trained_models/naive_bayes_model_v5_pneumonia.sav','rb'))

# need adjust model folder location according to uses

@st.cache_resource(ttl=3600,show_spinner="Loading image classification model...")
def load_img_cls_model():
    return pickle.load(open('C:/Users/A S U S/Desktop/med_vision/Pre-trained_models/image_classifying_preTrained_model.sav','rb'))
    # return pickle.load(open('Pre-trained_models/image_classifying_preTrained_model.sav','rb'))

image_classification_model = load_img_cls_model()

@st.cache_resource(ttl=3600,show_spinner="Loading CNN model...")
def load_cnn_p_model():
    return load_model('C:/Users/A S U S/Desktop/med_vision/Pre-trained_models/cnn_pneumonia_h5_v02.h5')
    # return load_model('Pre-trained_models/cnn_pneumonia_h5_v02.h5')


@st.cache_resource(ttl=3600,show_spinner="Loading Transfer Learning model...")
def load_tl_p_model():
    return pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/pneumonia-disease-pretrained-mode.sav','rb'))


@st.cache_resource(ttl=3600,show_spinner="Loading Transfer Learning model...")
def load_tl_eye_model():
    return pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/eye-disease-pretrained-mode.sav','rb'))


@st.cache_resource(ttl=3600,show_spinner="Loading CNN model...")
def load_cnn_eye_model():
    return pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/CNN-Eye_disease.sav','rb'))


@st.cache_resource(ttl=3600,show_spinner="Loading Naive Bayes model...")
def load_NB_p_model():
    return pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/naive_bayes_model_v5_pneumonia.sav','rb'))







# Pneumonia_model = pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/pneumonia-disease-pretrained-mode.sav','rb'))

# eye_disease_model = pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/eye-disease-pretrained-mode.sav','rb'))

# image_classification_model = pickle.load(open('C:/Users/A S U S/Desktop/med_vision/Pre-trained_models/image_classifying_preTrained_model.sav','rb'))

# cnn_pneumonia_model = load_model('C:/Users/A S U S/Desktop/med_vision/Pre-trained_models/cnn_pneumonia_h5_v02.h5')

# cnn_eye_disease_model = pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/CNN-Eye_disease.sav','rb'))

# naive_bayes_pneumonia_model = pickle.load(open('C:/Users/A S U S/MultiDisease_predict/Pre-trained_models/naive_bayes_model_v5_pneumonia.sav','rb'))




# models prediction function
# image classification model
def pre_img_class(image_path,model):
    individual_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    individual_image = Image.open(image_path).convert("RGB")
    individual_image = individual_image_transform(individual_image)
    individual_image = individual_image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    individual_image = individual_image.to(device)
    model.eval()

    with torch.no_grad():
        output = model(individual_image)

    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    return predicted_class

# pneumonia prediction by TL
def predict_pneumonia(image_path, model):
    # pass
    individual_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomGrayscale(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.0020], [0.0010])
    ])

    individual_image = Image.open(image_path).convert("RGB")
    individual_image = individual_image_transform(individual_image)
    individual_image = individual_image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    individual_image = individual_image.to(device)
    model.eval()

    with torch.no_grad():
        output = model(individual_image)

    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    return predicted_class



# eye-disease prediction using transfer learning
def predict_eye_disease_tf(image_path, model):
    # pass
    individual_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.0014, 0.0012, 0.0007], [0.0007, 0.0006, 0.0004])
    ])

    individual_image = Image.open(image_path)

    individual_image = individual_image_transform(individual_image)
    individual_image = individual_image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_trained_moedl = model.to(device)
    individual_image = individual_image.to(device)

    pre_trained_moedl.eval()
    with torch.no_grad():
        output = pre_trained_moedl(individual_image)
    
    _,predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    return predicted_class




def predict_pneumonia_using_CNN(image_path, model):
    img_size = 258
    img_path = image_path
    img = image.load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')
    # Convert grayscale image to RGB
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.concatenate([img_array] * 3, axis=-1)  # Repeat grayscale to all three channels
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction


def predict_eye_disease_using_CNN(image_path,model):
    # pass
    img_size=150
    img_path = image_path
    img = image.load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 

    # for matching the image shape
    img_array_rgb = np.concatenate([img_array, img_array, img_array], axis=-1)

    # Preprocess the image
    img_array_rgb /= 255.0
    prediction = model.predict(img_array_rgb)
    return prediction


def predict_pneumonia_using_Naive_Bayes(image_path,model):
    # pass
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_resized = img.resize((128, 128))  # Resize for consistency
    img_array = np.array(img_resized)
    img_flat = img_array.flatten()

    prediction = model.predict([img_flat])
    return prediction





# load models



def load_pneumonia_cnn_model():
    st.write("# Pneumonia CNN Model")
    # Load Kopil's CNN model and perform predictions
    btn=False
    btn_text="check prediction"
    chest_x_ray_image = st.file_uploader('Please upload your chest x-ray image', type=['jpeg', 'png', 'jpg'], on_change=change_photo_state)
    
    if chest_x_ray_image is not None:
        if st.session_state["photo"] == "done":
            text = "Image uploading in progress. Please wait..."
            progress_bar(text)
            st.image(chest_x_ray_image, caption='Uploaded Image.')
            st.session_state["photo"] ="image uploaded"
           
        # if st.session_state["check_button"] =="clicked":
        #     st.image(chest_x_ray_image, caption='Uploaded Image.')
        if st.session_state.clicked:
            time.sleep(3.5)
            st.image(chest_x_ray_image, caption='Uploaded Image.')
            btn = st.button("Check Prediction again")
        else:
            btn = st.button("Check Prediction",on_click=click_button)
        
        if st.session_state.clicked:
            predict_image_class = pre_img_class(chest_x_ray_image, image_classification_model)
            # checking image class whether it is chest X-ray image
            btn_text="check prediction again"
            text = "Operation in progress. Please wait for predicted value..."
            progress_bar(text)

            if predict_image_class == 0:
                # Predict the class Pneumonia or normal
                cnn_pneumonia_model = load_cnn_p_model()
                predicted_class = predict_pneumonia_using_CNN(chest_x_ray_image, cnn_pneumonia_model)
                
                if predicted_class[0][0] > 0.5:
                    st.error('The image is predicted as Pneumonia.')
                else:
                    st.success('The image is predicted as Normal.')
                st.snow()
            else:
                st.warning("Unable to determine the prediction. Because uploaded image is not chest X-ray image.")
    else:
        st.warning('Please upload a chest x-ray image.')






def load_pneumonia_transfer_learning_model():
    st.write("##### Transfer Learning Model useses DenseNet161 pretrained model")
    # st.write("### Right now We're unable to load this model.")
    # st.write("### Due to free low memory resources in cloud.")
    # st.write("### Stay tuned for our update.")
    # Load transfer learning model and perform predictions
    chest_x_ray_image = st.file_uploader('Please uploade your chest x-ray image',type=['jpeg','png','jpg'])
    if chest_x_ray_image is not None:
        st.image(chest_x_ray_image, caption='Uploaded Image.')
        # checking image class whether it is chest X-ray image
        predict_image_class = pre_img_class(chest_x_ray_image,image_classification_model)
        if(predict_image_class == 0):
            # Predict the class Pneumonia or normal
            Pneumonia_model = load_tl_p_model()
            predicted_class = predict_pneumonia(chest_x_ray_image, Pneumonia_model)
            if predicted_class == 0:
                st.success("The image is predicted as Normal.")
            elif predicted_class == 1:
                st.error("The image is predicted as Pneumonia.")
        else:
            st.warning("Unable to determine the prediction. Because uploaded image is not chest X-ray image.")
    else:
        st.warning('Please upload a chest x-ray image.')





def load_pneumonia_naive_bayes_model():
    st.write("# Pneumonia Naive Bayes Model")
    # st.write("### Right now We're unable to load this model.")
    # st.write("### Due to free low memory resources in cloud.")
    # st.write("### Stay tuned for our update.")
    # Load Kopil's CNN model and perform predictions
    btn=False
    btn_text="check prediction"
    chest_x_ray_image = st.file_uploader('Please upload your chest x-ray image', type=['jpeg', 'png', 'jpg'], on_change=change_photo_state)
    
    if chest_x_ray_image is not None:
        if st.session_state["photo"] == "done":
            text = "Image uploading in progress. Please wait..."
            progress_bar(text)
            st.image(chest_x_ray_image, caption='Uploaded Image.')
            st.session_state["photo"] ="image uploaded"
           
        # if st.session_state["check_button"] =="clicked":
        #     st.image(chest_x_ray_image, caption='Uploaded Image.')
        if st.session_state.clicked:
            time.sleep(2)
            st.image(chest_x_ray_image, caption='Uploaded Image.')
            btn = st.button("Check Prediction again")
        else:
            btn = st.button("Check Prediction",on_click=click_button)

        
        if st.session_state.clicked:
            predict_image_class = pre_img_class(chest_x_ray_image, image_classification_model)
            # checking image class whether it is chest X-ray image
            btn_text="check prediction again"
            text = "Operation in progress. Please wait for predicted value..."
            progress_bar(text)

            if predict_image_class == 0:
                # Predict the class Pneumonia or normal
                naive_bayes_pneumonia_model = load_NB_p_model()
                predicted_class = predict_pneumonia_using_Naive_Bayes(chest_x_ray_image, naive_bayes_pneumonia_model)
                
                if predicted_class[0] == 1:
                    st.error('The image is predicted as Pneumonia.')
                else:
                    st.success('The image is predicted as Normal.')
                st.snow()
            else:
                st.warning("Unable to determine the prediction. Because uploaded image is not chest X-ray image.")
    else:
        st.warning('Please upload a chest x-ray image.')





def load_eye_disease_cnn_model():
    st.write("##### CNN Model uses pretrained model")
    # st.write("### Right now We're unable to load this model.")
    # st.write("### Due to free low memory resources in cloud.")
    # st.write("### Stay tuned for our update.")
    # Loading cnn model and perform predictions
    eye_disease_image = st.file_uploader('Please uploade your eye-disease image',type=['jpeg','png','jpg'])
    if eye_disease_image is not None:
        st.image(eye_disease_image,caption='Uploaded image')

        # checking image class whether it is eye image or not
        predict_image_class = pre_img_class(eye_disease_image,image_classification_model)
        if predict_image_class == 1:
            cnn_eye_disease_model = load_cnn_eye_model()
            predicted_class = predict_eye_disease_using_CNN(eye_disease_image,cnn_eye_disease_model)
            if(predicted_class [0][0] > 0.5):
                st.error("The image is predicted as Cataract.")
            elif(predicted_class [0][1] > 0.5):
                st.error("The image is predicted as Diabetic Retinopethy.")
            elif(predicted_class [0][2] > 0.5):
                st.error("The image is predicted as Glaucoma.")
            elif(predicted_class [0][3] > 0.5):
                st.success('The image is predicted as Normal.')
        else:
            st.warning("Unable to determine the prediction. Because uploaded image is not eye image.")
    else:
        st.warning('Please upload a eye-disease image.')




def load_eye_disease_transfer_learning_model():
    st.write("##### Transfer Learning Model uses DenseNet161 pretrained model")
    # st.write("### Right now We're unable to load this model.")
    # st.write("### Due to free low memory resources in cloud.")
    # st.write("### Stay tuned for our update.")
    # Load your transfer learning model and perform predictions
    eye_disease_image = st.file_uploader('Please uploade your eye-disease image',type=['jpeg','png','jpg'])
    if eye_disease_image is not None:
        st.image(eye_disease_image,caption='Uploaded image')

        # checking image class whether it is chest X-ray image
        predict_image_class = pre_img_class(eye_disease_image,image_classification_model)
        if predict_image_class == 1:
            eye_disease_model = load_tl_eye_model()
            predicted_class = predict_eye_disease_tf(eye_disease_image,eye_disease_model)
            if(predicted_class == 0):
                st.error("The image is predicted as Cataract.")
            elif(predicted_class == 1):
                st.error("The image is predicted as Diabetic Retinopethy.")
            elif(predicted_class == 2):
                st.error("The image is predicted as Glaucoma.")
            elif(predicted_class == 3):
                st.success('The image is predicted as Normal.')
        else:
            st.warning("Unable to determine the prediction. Because uploaded image is not eye image.")
        
    else:
        st.warning('Please upload a eye-disease image.')



def load_eye_disease_naive_bayes_model():
    st.write("# Eye Disease Naive Bayes Model")
    st.write("### Right now We're unable to load this model.")
    st.write("### Due to free low memory resources in cloud.")
    st.write("### Stay tuned for our update.")
    # Load Kopil's your naive bayes model and perform predictions
    pass

# navigation bar
# home page's code
# styling title
st.markdown("<style>h2{ color: white;text-shadow: 2px 2px 5px #000000;letter-spacing: 3px;word-spacing: 10px; }</style>", unsafe_allow_html=True)
st.markdown("<u><h2>Pneumonia and Eye Disease Recognition System</h2></u>", unsafe_allow_html=True)

selected_disease = st.sidebar.selectbox('Select Disease', ['Pneumonia', 'Eye disease'])

# Load models based on disease selection
if selected_disease == 'Pneumonia':
    selected_model = st.sidebar.selectbox('Select Model', ['CNN', 'Transfer Learning', 'Naive Bayes'])
    st.subheader(f'{selected_disease} - {selected_model} Prediction Page')

    if selected_model == 'CNN':
        load_pneumonia_cnn_model()
    elif selected_model == 'Transfer Learning':
        load_pneumonia_transfer_learning_model()
    elif selected_model == 'Naive Bayes':
        load_pneumonia_naive_bayes_model()

elif selected_disease == 'Eye disease':
    selected_model = st.sidebar.selectbox('Select Model', ['CNN', 'Transfer Learning', 'Naive Bayes'])
    st.subheader(f'{selected_disease} - {selected_model} Prediction Page')

    if selected_model == 'CNN':
        load_eye_disease_cnn_model()
    elif selected_model == 'Transfer Learning':
        load_eye_disease_transfer_learning_model()
    elif selected_model == 'Naive Bayes':
        load_eye_disease_naive_bayes_model()



