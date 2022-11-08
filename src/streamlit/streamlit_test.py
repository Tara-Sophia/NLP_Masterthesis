# -*- coding: utf-8 -*-
# import os

# import matplotlib.pyplot as plt
# import seaborn as sns
# from predict_lr_new import *

# import streamlit as st

# # importing all the helper fxn from predict_lr_new.py


# # from helper file we are importing load_model, predict_probability, top_symptoms, lime_explainer
# file_path = os.path.join(
#     "models", "clf", "sklearn_logistic_regression_model.pkl"
# )
# model = load_model(file_path)

# st.title("Welcome to medical symptom checker")

# tab1, tab2, tab3, tab4 = st.tabs(
#     [
#         "Speech to text",
#         "Transcription",
#         "Classification",
#         "Probabilities",
#     ]
# )

# with tab1:
#     st.write("Coming soon")

# with tab2:
#     with st.form("inputfield", clear_on_submit=True):
#         transcription = st.text_area(
#             "Please describe how you are feeling"
#         )

#         submit = st.form_submit_button("Submit")
#         if submit:
#             st.write("Your text: ", transcription)
#             keywords = [
#                 "cardiac",
#                 "ventricular",
#                 "mitral",
#                 "left",
#                 "valve",
#             ]
#             st.write("These are the most important keywords")
#             st.write(transcription)


# with tab3:
#     prediction = predict_probability(model, [transcription])
#     prob1 = prediction["Probability"][0]
#     prob2 = prediction["Probability"][1]
#     prob3 = prediction["Probability"][2]

#     # find the value of top_symptoms(model) where index is prediction.index[0]
#     top_symptoms = top_symptoms(model)
#     top_symptoms_1 = top_symptoms[prediction.index[0]]
#     top_symptoms_2 = top_symptoms[prediction.index[1]]
#     top_symptoms_3 = top_symptoms[prediction.index[2]]

#     # find the value of top symptoms from lime
#     feat_importance = lime_explainer(model, transcription)
#     top_symptoms_1_lime = get_words(
#         prediction.index[0], feat_importance
#     )
#     top_symptoms_2_lime = get_words(
#         prediction.index[1], feat_importance
#     )
#     top_symptoms_3_lime = get_words(
#         prediction.index[2], feat_importance
#     )

#     st.subheader(
#         "Based on our algorithm you should consider contacting these departments"
#     )
#     with st.expander(prediction.index[0]):
#         st.metric(
#             label="Percentage of probability",
#             value="{:.0%}".format(prob1),
#         )
#         st.write(
#             "Decision was based on these symptoms from your description:"
#         )
#         st.write(top_symptoms_1_lime)
#         st.write(
#             "Most relevant symptoms for this department in general:"
#         )
#         st.write(top_symptoms_1)

#     with st.expander(prediction.index[1]):
#         st.metric(
#             label="Percentage of probability",
#             value="{:.0%}".format(prob2),
#         )
#         st.write(
#             "Decision was based on these symptoms from your description:"
#         )
#         st.write(top_symptoms_2_lime)
#         st.write(
#             "Most relevant symptoms for this department in general:"
#         )
#         st.write(top_symptoms_2)

#     with st.expander(prediction.index[2]):
#         st.metric(
#             label="Percentage of probability",
#             value="{:.0%}".format(prob3),
#         )
#         st.write(
#             "Decision was based on these symptoms from your description:"
#         )
#         st.write(top_symptoms_3_lime)
#         st.write(
#             "Most relevant symptoms for this department in general:"
#         )
#         st.write(top_symptoms_3)

# with tab4:
#     prediction = predict_probability(model, [transcription])
#     st.subheader("Probability of each department")
#     st.write(prediction)
