import sys
import os
import cv2
import numpy as np
import joblib
import data_pre_processing as prep_data
import training


SVM_model = joblib.load('model/SVM_text_clf.pkl')
logis_model = joblib.load('model/logisticRegr_text_clf.pkl')
dec_model = joblib.load('model/decision_text_clf.pkl')

test_list = []
# test_list.append("insert text") 

count_v = training.getCountVectorModel()
tf_transformer = training.getTFtranformer()

test_c = count_v.transform(test_list)
test_t = tf_transformer.transform(test_c)
dec_pred_result = dec_model.predict(test_t)

for x in dec_pred_result:
    if x == 0:

        print("Not a Misogynistic Tweet")
    else:
        print("Misogynistic Tweet")