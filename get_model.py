import requests
import os


def download_file_from_google_drive(drive_file_id, output_dir):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': drive_file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': drive_file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, output_dir)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, output_dir):
    chunk_size = 32768

    with open(output_dir, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_model():
    file_id = '1-7gn_dEEakB3rOCymuBz32qY0OsDUYPK'
    model_name = 'model.pth'
    if model_name not in os.listdir():
        download_file_from_google_drive(file_id, model_name)
