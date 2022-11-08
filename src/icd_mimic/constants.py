import os

ICD9_SPECIALTY_DICT = {
    "Certain Conditions Originating In The Perinatal Period": "Obstetrics & Gynaecology",
    "Complications Of Pregnancy, Childbirth, And The Puerperium": "Obstetrics & Gynaecology",
    "Congenital Anomalies": "Primary Care",
    "Diseases Of The Blood And Blood-Forming Organs": "Hematology",
    "Diseases Of The Circulatory System": "Cardiothoracic & Vascular",
    "Diseases Of The Digestive System": "Gastroenterology",
    "Diseases Of The Genitourinary System": "Urology",
    "Diseases Of The Musculoskeletal System And Connective Tissue": "Orthopedic surgery",
    "Diseases Of The Nervous System And Sense Organs": "Neurology",
    "Diseases Of The Respiratory System": "Pulmonology",
    "Diseases Of The Skin And Subcutaneous Tissue": "Dermatology",
    "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders": "Endocrinology",
    "Infectious And Parasitic Diseases": "Infectious Disease Specialty",
    "Injury And Poisoning": "Emergency Department",
    "Mental Disorders": "Psychiatry",
    "Neoplasms": "Oncology",
    "Supplementary Classification Of External Causes Of Injury And Poisoning": "Emergency Department",
    "Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services": "Internal Medicine Department",
    "Symptoms, Signs, And Ill-Defined Conditions": "Internal Medicine Department",
}


NOTEEVENTS_COLS = [
    "ROW_ID",
    "SUBJECT_ID",
    "HADM_ID",
    "CATEGORY",
    "ISERROR",
    "TEXT",
]

MIMIC_DIAGNOSES_CSV = os.path.join(
    "data", "raw", "mimic_iii", "DIAGNOSES_ICD.csv"
)
ICD9_CSV = os.path.join(
    "data", "interim", "icd", "icd9_codes_and_des.csv"
)
MIMIC_NOTEEVENTS_CSV = os.path.join(
    "data", "raw", "mimic_iii", "NOTEEVENTS.csv"
)

DIAGNOSES_NOTEEVENTS_CSV = os.path.join(
    "data", "processed", "mimic_iii", "diagnoses_noteevents.csv"
)
