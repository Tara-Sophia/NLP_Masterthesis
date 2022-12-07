# -*- coding: utf-8 -*-
"""
Description:
    This script contains all constants used in the project.
"""

import os

ICD9_SPECIALTY_DICT = {
    "Certain Conditions Originating In The Perinatal Period": "Obstetrics & Gynaecology",  # noqa: E501
    "Complications Of Pregnancy, Childbirth, And The Puerperium": "Obstetrics & Gynaecology",  # noqa: E501
    "Congenital Anomalies": "Primary Care",  # noqa: E501
    "Diseases Of The Blood And Blood-Forming Organs": "Hematology",  # noqa: E501
    "Diseases Of The Circulatory System": "Cardiothoracic & Vascular",  # noqa: E501
    "Diseases Of The Digestive System": "Gastroenterology",  # noqa: E501
    "Diseases Of The Genitourinary System": "Urology",  # noqa: E501
    "Diseases Of The Musculoskeletal System And Connective Tissue": "Orthopedic surgery",  # noqa: E501
    "Diseases Of The Nervous System And Sense Organs": "Neurology",  # noqa: E501
    "Diseases Of The Respiratory System": "Pulmonology",  # noqa: E501
    "Diseases Of The Skin And Subcutaneous Tissue": "Dermatology",  # noqa: E501
    "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders": "Endocrinology",  # noqa: E501
    "Infectious And Parasitic Diseases": "Infectious Disease Specialty",  # noqa: E501
    "Injury And Poisoning": "Emergency Department",  # noqa: E501
    "Mental Disorders": "Psychiatry",  # noqa: E501
    "Neoplasms": "Oncology",  # noqa: E501
    "Supplementary Classification Of External Causes Of Injury And Poisoning": "Emergency Department",  # noqa: E501
    "Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services": "Internal Medicine Department",  # noqa: E501
    "Symptoms, Signs, And Ill-Defined Conditions": "Internal Medicine Department",  # noqa: E501
}


NOTEEVENTS_COLS = [
    "ROW_ID",
    "SUBJECT_ID",
    "HADM_ID",
    "CATEGORY",
    "ISERROR",
    "TEXT",
]

MIMIC_DIAGNOSES_CSV = os.path.join("data", "raw", "mimic_iii", "DIAGNOSES_ICD.csv")
ICD9_CSV = os.path.join("data", "interim", "icd", "icd9_codes_and_des.csv")
MIMIC_NOTEEVENTS_CSV = os.path.join("data", "raw", "mimic_iii", "NOTEEVENTS.csv")

DIAGNOSES_NOTEEVENTS_CSV = os.path.join(
    "data", "processed", "mimic_iii", "diagnoses_noteevents.csv"
)

MIMIC_PERSONALIZED_STOPWORDS_FILTERED = [
    "He",
    "She",
    "patient",
    "**]",
    "[**Hospital1",
    "The",
    "given",
    "showed",
    "also",
    "In",
    "On",
    "denies",
    "history",
    "found",
    "transferred",
    "ED",
    "Patient",
    "Name",
    "noted",
    "s/p",
    "started",
    "prior",
    "18**]",
    "admitted",
    "CT",
    "Pt",
    "2",
    "presented",
    "IV",
    "reports",
    "pt",
    "recent",
    "last",
    "received",
    "No",
    "BP",
    "ED,",
    "year",
    "old",
    "[**Known",
    "past",
    "1",
    "days",
    "lastname",
    "His",
    "OSH",
    "arrival",
    "time",
    "[**Last",
    "yo",
    "This",
    "presents",
    "well",
    "[**Hospital",
    "HR",
    "male",
    "mg",
    "x",
    "day",
    "Her",
    "admission",
    "without",
    "At",
    "home",
    "felt",
    "initial",
    "developed",
    "revealed",
    "(un)",
    "3",
    "since",
    "placed",
    "increased",
    "per",
    "A",
    "h/o",
    "recently",
    "CXR",
    "Per",
    "severe",
    "significant",
    "treated",
    "w/",
    "transfer",
    "L",
    "underwent",
    "initially",
    "[**Hospital3",
    "due",
    "states",
    "Denies",
    "one",
    "R",
    "notable",
    "symptoms",
    "seen",
    "ED.",
    "O2",
    "called",
    "RR",
    "status",
    "EKG",
    "several",
    "review",
    "Of",
    "feeling",
    "continued",
    "fevers,",
    "hospital",
    "[**Location",
    "(NI)",
    "Mr.",
    "went",
    "HTN,",
    "T",
    "(STitle)",
    "note,",
    "today",
    "VS",
    "became",
    "discharged",
    "MICU",
    "weeks",
    "ago",
    "episode",
    "4",
    "taken",
    "new",
    "sent",
    "normal",
    "[**Name",
    "medical",
    "episodes",
    "two",
    "chills,",
    "aortic",
    "100%",
    "denied",
    "improved",
    "possible",
    "unable",
    "SOB",
    "EMS",
    "morning",
    "associated",
    "elevated",
    "large",
    "reported",
    "brought",
    "week",
    "[**First",
    "RA.",
    "night",
    "course",
    "Dr.",
    "M",
    "GI",
    "decreased",
    "ICU",
    "WBC",
]
