import tkinter as tk
import requests
import sys
import os
import cv2
import numpy as np
import joblib
import data_pre_processing as prep_data
import training

SVM_model = joblib.load('model/SVM_text_clf.pkl')
NB_model = joblib.load('model/NB_text_clf.pkl')
logis_model = joblib.load('model/logisticRegr_text_clf.pkl')

count_v = training.getCountVectorModel()
tf_transformer = training.getTFtranformer()

root = tk.Tk()
root.geometry("800x500")

label = tk.Label(root, text = "Misogynyst Text Clussification", font=('THSarabunNew', 20, 'bold'))
label.pack()

# label = tk.Label(text="I'm in Frame A")
# label.pack()

frame = tk.Frame(root, bg='#933DB5', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

# ช่อง input
entry = tk.Text(frame, font=('THSarabunNew', 18))
entry.place(relwidth=0.65, relheight=1)

# ปุ่มตกลง
button = tk.Button(frame, text='ตรวจสอบ', font=('THSarabunNew', 18), command=lambda: get_data(entry.get("1.0","end-1c")))
button.place(relx=0.7, relwidth=0.3, relheight=1)

# โชว์ขผลลัพธ์
lower_frame = tk.Frame(root, bg='#933DB5', bd=10)
lower_frame.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.6, anchor='n')
label = tk.Label(lower_frame, font=('THSarabunNew', 18), anchor='nw', justify='left')

label.place(relwidth=1, relheight=1)

def get_data(text):
    
    test_list = []
    test_list.append(text)
    test_c = count_v.transform(test_list)
    test_t = tf_transformer.transform(test_c)
    svm_pred_result = SVM_model.predict(test_t)

    for x in svm_pred_result:
        if x == 0:
            label.config(text="Not a Misoginistic Tweet")
            print("Not a Misoginistic Tweet")
        else:
            label.config(text="Misoginistic Tweet")
            print("Misoginistic Tweet")


root.mainloop()