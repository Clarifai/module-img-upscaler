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
def get_upscaled_img(img_file, upscale_percent):
    image_b = img_file.getvalue()
    response = Model(model_url).predict_by_bytes(image_b, "image")
    img = response.outputs[0].data.image.base64
    img_info = json_format.MessageToDict(response.outputs[0].data.image.image_info)
    return img, img_info


def upload_image(img_bytes):
    img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    Inputs(user_id=auth.user_id, app_id=auth.app_id).upload_from_bytes(img_id, img_bytes)
    st.success("Image upscaled & uploaded successfully!")


with st.form(key="upscaler_form"):
    img_file = st.file_uploader("Select an image", type=["png", "jpg", "jpeg"])
    upscale_percent = None
    # upscale_percent = st.slider("Upscaling percentage:", min_value=110, max_value=500, value=110, step=10, format="%d")
    submit_button = st.form_submit_button(label="Upscale & Upload")

if submit_button and img_file:
    with st.spinner('Wait for it...'):
        img, img_info = get_upscaled_img(img_file, upscale_percent)
    st.write(f"Image Upscaled to {img_info['width']}x{img_info['height']} (ht x wt)")
    st.image(img, caption="Upscaled Image", use_column_width=True)
    upload_image(img)
