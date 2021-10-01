# App to serve the Layoutlm model using Streamlit
from pathlib import Path
import urllib
import tempfile

from PIL import Image, ImageDraw, ImageFont
from numpy.core.fromnumeric import size
import streamlit as st

from app_engine import AppEngine

DEFAULT_DATA_BASE_DIR = 'data'
IMAGE_DIR = 'demo_scan'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE]

LABELS = ['O', 'B-MENU.CNT', 'B-MENU.DISCOUNTPRICE', 'B-MENU.ETC', 'B-MENU.ITEMSUBTOTAL', 'B-MENU.NM', 'B-MENU.NUM', 'B-MENU.PRICE', 'B-MENU.SUB_CNT', 'B-MENU.SUB_ETC', 'B-MENU.SUB_NM', 'B-MENU.SUB_PRICE', 'B-MENU.SUB_UNITPRICE', 'B-MENU.UNITPRICE', 'B-MENU.VATYN', 'B-SUB_TOTAL.DISCOUNT_PRICE', 'B-SUB_TOTAL.ETC', 'B-SUB_TOTAL.OTHERSVC_PRICE', 'B-SUB_TOTAL.SERVICE_PRICE', 'B-SUB_TOTAL.SUBTOTAL_PRICE', 'B-SUB_TOTAL.TAX_PRICE', 'B-TOTAL.CASHPRICE', 'B-TOTAL.CHANGEPRICE', 'B-TOTAL.CREDITCARDPRICE', 'B-TOTAL.EMONEYPRICE', 'B-TOTAL.MENUQTY_CNT', 'B-TOTAL.MENUTYPE_CNT', 'B-TOTAL.TOTAL_ETC', 'B-TOTAL.TOTAL_PRICE', 'B-VOID_MENU.NM', 'B-VOID_MENU.PRICE', 'I-MENU.CNT', 'I-MENU.DISCOUNTPRICE', 'I-MENU.ETC', 'I-MENU.NM', 'I-MENU.PRICE', 'I-MENU.SUB_ETC', 'I-MENU.SUB_NM', 'I-MENU.UNITPRICE', 'I-MENU.VATYN', 'I-SUB_TOTAL.DISCOUNT_PRICE', 'I-SUB_TOTAL.ETC', 'I-SUB_TOTAL.OTHERSVC_PRICE', 'I-SUB_TOTAL.SERVICE_PRICE', 'I-SUB_TOTAL.SUBTOTAL_PRICE', 'I-SUB_TOTAL.TAX_PRICE', 'I-TOTAL.CASHPRICE', 'I-TOTAL.CHANGEPRICE', 'I-TOTAL.CREDITCARDPRICE', 'I-TOTAL.EMONEYPRICE', 'I-TOTAL.MENUQTY_CNT', 'I-TOTAL.MENUTYPE_CNT', 'I-TOTAL.TOTAL_ETC', 'I-TOTAL.TOTAL_PRICE', 'I-VOID_MENU.NM']

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/pierre-si/layoutlm/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@st.cache(allow_output_mutation=True)
def load_model():
    handle = AppEngine()
    return handle


def run_app(img):
    display_image = Image.open(img)

    left_column, right_column = st.columns(2)
    left_column.image(display_image, caption = "Selected Input")

    handle = load_model()
    print(img)
    res = handle(img)

    # OCR bounding boxes
    read = res["easyocr"]
    ocr_image = display_image.copy()
    draw = ImageDraw.Draw(ocr_image, "RGBA")
    bboxes = []
    for segment in read:
        bbox = segment[0][0]+segment[0][2]
        if segment[2] > 0.1:
            bboxes.append(bbox)
        draw.rectangle(bbox, outline="darkblue" if segment[2] > 0.1 else "grey", width=7)

    right_column.image(ocr_image, caption = "OCR results (blue: above confidence threshold, grey: below)")


    # LayoutLM predictions
    draw = ImageDraw.Draw(display_image, "RGBA")
    font = ImageFont.truetype("arial.ttf", 20)
    label2color = {'MENU':'blue', "SUB_TOTAL":'green', 'TOTAL':'orange', }

    # bboxes = res["x"]["bboxes"][0]
    # tokens = res["x"]["tokens"][0]
    y = res["class_index"]
    for box, pred in zip(bboxes, y):
        label = LABELS[pred]
        draw.rectangle(box, outline=label2color[label.split(".")[0][2:]], width=1)
        draw.text((box[0] - 30, box[1] - 10), label, fill=label2color[label.split(".")[0][2:]], font=font)
    st.image(display_image, caption = "Classification results")


def main():

    st.sidebar.title("Explore the Following")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.sidebar.write("More About The Project")
        st.sidebar.write("Hi there! If you want to check out the source code, please visit my github repo: https://github.com/pierre-si/layoutlm/")

        st.write(get_file_content_as_string("README.md"))

    elif app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        st.sidebar.write(" ------ ")

        directory = Path(DEFAULT_DATA_BASE_DIR) / IMAGE_DIR

        scans = []
        for filepath in directory.iterdir():
            # Find all valid images
            # if imghdr.what(filepath) is not None:
            scans.append(filepath.name)

        scans.sort()

        option = st.sidebar.selectbox('Please select a sample image, then click Magic Time button', scans)

        display_image = Image.open(Path(directory)/option)
        preview = st.image(display_image)

        pressed = st.sidebar.button('Magic Time')
        if pressed:
            preview.empty()
            st.empty()
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')

            pic = Path(directory) / option

            run_app(pic)

    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
        st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
            and discarded after the final results are displayed. ')
        f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif'])
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=True)
            tfile.write(f.read())
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
            run_app(tfile.name)

main()
