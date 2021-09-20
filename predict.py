import pickle
import numpy as np
import pandas as pd

input = {
    "person_age": 30,
    "person_income": 5000,
    "person_home_ownership": "RENT",
    "person_emp_length": 6,
    "loan_intent": "MEDICAL",
    "loan_grade": "A",
    "loan_amnt": 6000,
    "loan_int_rate": 7,
    "loan_percent_income": 0.15,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 2
}

def preprocess_input(input_data):
    """ preprocess input data """
    # mapper_replace = {
    #     "null": np.nan,
    #     None: np.nan
    # }
    input_data.pop("loan_status", None)
    input_data.pop("score_proba", None)
    input_data = pd.DataFrame.from_dict(input_data, orient='index')#.replace(mapper_replace)
    return input_data


def make_predictions(input_data):
    """ function to make final prediction using pipeline """
    with open('trained_model/FE-SC-IMP-OHE-1.0.0.pkl', 'rb') as f:
        fe = pickle.load(f)

    with open('trained_model/M-LR-1.0.0.pkl', 'rb') as f:
        model = pickle.load(f)
    input_data = preprocess_input(input_data).T.replace({
        None: np.nan,
        "null": np.nan,
        "": np.nan
        })
    
    # Feature Engineering with pipeline
    input_data = fe.transform(input_data)

    # model prediction
    result = model.predict_proba(input_data)[:, 1]
    return result 
    
if __name__ == "__main__":
    # data = preprocess_input(input)
    result = make_predictions(input)
    print(result)