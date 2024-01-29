import logging
import os
import json
from datetime import datetime

import dill
import pandas as pd
from pydantic import BaseModel

path = os.environ.get('PROJECT_PATH', '.')

class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


def load_model():
    with open(path + '/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    return model


def load_data() -> list:
    data_dir = path + '/data/test/'
    json_files = [os.path.abspath(data_dir + x) for x in os.listdir(data_dir)]
    return json_files


def make_predict(form: Form, model) -> str:
    df = pd.DataFrame.from_dict([form])
    y = model.predict(df)

    return f'{df.id[0]}, {y[0]}\n'

def predict():
    model = load_model()
    test_data = load_data()
    preds_file_name = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'

    if not test_data:
        logging.info('Test data is empty')
    else:
        prediction = ''
        for file in test_data:
            f = open(file)
            json_data = json.load(f)
            prediction = prediction + make_predict(json_data, model)

        with open(preds_file_name, 'wb') as pred_file:
            pred_file.write(prediction.encode('utf-8'))

if __name__ == '__main__':
    predict()
