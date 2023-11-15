import streamlit as st
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from google.protobuf import json_format
import random, string
import os


st.set_page_config(layout="centered")

ClarifaiStreamlitCSS.insert_default_css(st)
auth = ClarifaiAuthHelper.from_streamlit(st)
if not os.environ["CLARIFAI_PAT"]:
    os.environ['CLARIFAI_PAT'] = auth._pat

model_url = "https://clarifai.com/stability-ai/Upscale/models/stabilityai-upscale"

st.title("Image Upscaler")

st.subheader("Choose an image to get started")

@st.cache_data
def get_upscaled_img(img_file, upscale_width):
    image_b = img_file.getvalue()
    inference_params = dict(width=upscale_width)
    # response = Model(model_url).predict_by_bytes(image_b, "image")
    response = Model(model_url).predict_by_bytes(image_b, "image", inference_params=inference_params)
    img = response.outputs[0].data.image.base64
    img_info = json_format.MessageToDict(response.outputs[0].data.image.image_info)
    return img, img_info


def upload_image(img_bytes):
    img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    Inputs(user_id=auth.user_id, app_id=auth.app_id).upload_from_bytes(img_id, img_bytes)
    st.success("Image upscaled & uploaded successfully!")


with st.form(key="upscaler_form"):
    img_file = st.file_uploader("Select an image", type=["png", "jpg", "jpeg"])
    upscale_width = st.number_input("Upscaling width: (Min:512 | Max:2048)", min_value=512, max_value=2048, value=1024, step=2, format="%d")
    submit_button = st.form_submit_button(label="Upscale & Upload")

if submit_button and img_file:
    with st.spinner('Wait for it...'):
        img, img_info = get_upscaled_img(img_file, upscale_width)
    st.write(f"Image Upscaled to {img_info['width']}x{img_info['height']} (ht x wt)")
    st.image(img, caption="Upscaled Image", use_column_width=True)
    upload_image(img)
