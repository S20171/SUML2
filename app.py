# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model


def main():

	st.set_page_config(page_title="Zdrowie app")
	overview = st.container()
	left= st.container()
	prediction = st.container()

	st.image("https://th.bing.com/th/id/R.d76bcbd618c2f70af30d167bc929227d?rik=XL3r6HV%2fi6oqnA&pid=ImgRaw&r=0")

	with overview:
		st.title("Zdrowie app")

	with left:
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=100)
		diseases_slider = st.slider("Choroby współistniejące", min_value=0, max_value=10)
		height_slider = st.slider("Wzrost", min_value=0, max_value=200)
		medicines_slider = st.slider("Leki", min_value=0, max_value=20, step=1)


	data = [[age_slider, age_slider, diseases_slider, height_slider, medicines_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba będzie mieć chorbę serca?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
