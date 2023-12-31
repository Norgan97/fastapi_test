import requests
import streamlit as st
import json

def main():

    st.title("Image classification")

    image = st.file_uploader("Choose an image", type=['jpg', 'jpeg'])

    if st.button("Classify!") and image is not None:
        st.image(image)
        files = {"file": image.getvalue()}
        res = requests.post("http://127.0.0.1:8000/classify", files=files)
        st.write(json.loads(res.text)['prediction'])

if __name__ == '__main__':
    main()