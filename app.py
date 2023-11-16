import streamlit as st
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_utils.auth.helper import ClarifaiAuthHelper
import random, string
import os
import io
from PIL import Image, ImageChops


st.set_page_config(layout="centered")

ClarifaiStreamlitCSS.insert_default_css(st)
auth = ClarifaiAuthHelper.from_streamlit(st)
if not os.environ["CLARIFAI_PAT"]:
    os.environ['CLARIFAI_PAT'] = auth._pat

model_url = "https://clarifai.com/stability-ai/Upscale/models/stabilityai-upscale"

st.title("Image Upscaler")

st.subheader("Choose an image to get started")

@st.cache_data
def get_upscaled_img(image_b, upscale_width):
    inference_params = dict(width=upscale_width)
    response = Model(model_url).predict_by_bytes(image_b, "image", inference_params=inference_params)
    up_img = Image.open(io.BytesIO(response.outputs[0].data.image.base64))
    return up_img


def upload_image(img_bytes):
    img_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    Inputs(user_id=auth.user_id, app_id=auth.app_id).upload_from_bytes(img_id, img_bytes)
    st.success("Image upscaled & uploaded successfully!")


def trim(im, org_shape):
    if org_shape[0] == org_shape[1]:
        return im
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


with st.form(key="upscaler_form"):
    img_file = st.file_uploader("Select an image", type=["png", "jpg", "jpeg"])
    upscale_width = st.number_input("Upscaling width: (Min:512 | Max:2048)", min_value=512, max_value=2048, value=1024, step=2, format="%d")
    submit_button = st.form_submit_button(label="Upscale & Upload")

if submit_button and img_file:
    with st.spinner('Upscaling...'):
        image_b = img_file.getvalue()
        org_img = Image.open(io.BytesIO(image_b))
        ups_img = get_upscaled_img(image_b, upscale_width)
        ups_img = trim(ups_img, org_img.size)
    st.write(f"Image Upscaled to {ups_img.size[0]}x{ups_img.size[1]} (wt x ht)")
    img_byte_arr = io.BytesIO()
    ups_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    st.image(ups_img, caption="Upscaled Image", use_column_width=True)
    with st.spinner('Uploading...'):
        upload_image(img_byte_arr)
