#import whisper
import os
#from pytube import YouTube, Playlist
import streamlit as st
import openai
# import tensorflow as tf
# import tensorflow_hub as hub
import pandas as pd
# import geopandas as gp
# import h3
import openai
# from shapely.geometry import Polygon
from openai import OpenAI
import os
import tiktoken
#import moviepy.editor as mp
#import re
#import random
#import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
#import io
#import markdown2
#import textwrap
from PIL import Image
from fuzzywuzzy import fuzz
from io import BytesIO
import logging

# Read file from GCS
#to upload the noisy hdbscan clusterer to GCP
from google.oauth2 import service_account

from google.cloud import storage

#upload the parquet file to GCP
service_account_info = {

"type" : "service_account",

"project_id": "utility-braid-351906",

"private_key_id": "bde1a70eb39a3564ffdd12141bc1bc9d661a5d1c",

"private_key":"-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDQgDLWhbjv8zZX\nHLLfTH4RN9YfMKScMsKjf0X0NrxDj6XHV0GZjjQFl9Laovktffk1uc2FVRFDjQi0\nxUIJ8oTj6/PY9zBhnY57lgbaMPS0BGh02hWsKdhMDZKD2G0guuRYjsUFtHgJcG+A\ngtJGIOG1fhk6LSl3EPN4zbH5vLj9wElLa7lS6Im8NjVL2AqDst+EDLYEv7hwlFxp\n2RK4vw+Cscm3IuiiQW1ZakYtLdlpWVl6Y1NVVEYqtK4XWLUsJdapErumWsZt4Sc8\nEBCnPWKuA7/9NQt8UJdjGhgIWPMVLmeusnDeUH6kzsAEIMUxnGL8LRcgfq1ipsvV\nrtSe88EbAgMBAAECggEAAIbKWhNVX7nAAeFPLFQvn1CCSGrhh4lzR2FrZk37FLqY\noxO71SzwQbFb9XRi57QBPJr/A16vH/oLsxUqcQb9QewXHg2lwCFbOkGrO8LS4199\negoQ8vHaG3sPSyjxQZtnNzgJFFYb1X5EsdDiwg2UNFagrq5tg+D1lHEEiYpp9y4T\ndk7khDPKt5wY7yu6MebCveZEZ+yzGfjAXoTkN5ZvVGAJAOTJK5D9rwEF2X6gbbC+\nwx+HD6QxJPXiCRaN4mGSPz/KpS8QrQkl0d1fUHBaRwTMIdBWnKnF4D6zf/l1EJLC\nCohOS6imGmsRlBYg4UJ11PdriWuZFwUNbhcm/oxT4QKBgQDxR4NYjd+Qu84ohn7d\nrKOv4KbQWSWbCwwU8DOkDjGT7Hv6EBlNVgvR5ZfeHqvkuLBa9+4LIyqp9IyiYnar\nbWmM7ms6nfwKEGKhrw4y9d1UQh/d4TDTBNIZsuwIiQoK0v63dNZ4jyJRMq2eM+wA\n7sTfIvz1sAVFgg/fozAqniakIQKBgQDdOLl4mT88cNaUehPGjOFVSasILV9LiVZA\n3o9pSN4LvFERgLFWsUcCUwTnpJYu+qOnUUjpk/6hY77eHrOtCQ1MU7sArxoebsg4\nMKyDfthf1C9xytqSLjKiyM9TSTxI41SioPXCOZxe1y5JBHptCds9lLzjCizSbSFx\nAN5vr949uwKBgBP62wqTPQcsNicu9ASBTlC7JrUsHKwZHxgAyBX2wu4/8AhGGwJH\ndNUd0RSor41SKfBuhXzQnbDTOm4b/z204r+z4pdJC9z9fF1tNJzNtVVL4H2sLzHa\nPVe5dEhEqNs6m7Mvbq8vEyVsL+pg3FM7cnwT1qS1vcoCujPHvK5ayFJhAoGASN0w\nKcrAC1ZXNxxmexVX+tGC5fSb2LNpl4A22ETJ7i+evBcZUiad7uQNT4bkeKDRWoDp\ndRXr3piN+3c9UxcSLDu/8l+6SJ/QjsFpcP5MonOvFNnt2AwjXX6q2xHaK1/FNrOx\nfsGfAZX6hs5UzKlcbxIYjOeDD+QmCaRxn3PbzZUCgYAy1Sa7cWQv2BLToSyH3/U7\ngNziYjaK4WuUFRmizWJYMjHjwiySimma+32iffS5LgvDCt1p4ulMm5Bo/7R27rjr\n606lUlS/58eTKnn4ERPhUiR6R81pC8VWFvQ5deeXDdu2rmi8igDzp37NF20tpMMR\nIKNXWVIknD8Awl8xhE6oTw==\n-----END PRIVATE KEY-----\n",

"client_email": "psil-app@utility-braid-351906.iam.gserviceaccount.com",

"client_id": "109250145406396845975",

"auth_uri":"https://accounts.google.com/o/oauth2/auth",

"token_uri": "https://oauth2.googleapis.com/token",

"auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",

"client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/psil-app%40utility-braid-351906.iam.gserviceaccount.com",

"universe_domain":"googleapis.com"

}

#Create credentials object
credentials = service_account.Credentials.from_service_account_info(service_account_info)


# This loads a csv from a gcs bucket and returns a pandas dataframe
def load_csv_from_gcs(bucket_name,filename):

# request_json = request.get_json(silent=True)

    client = storage.Client(credentials=credentials)

    bucket = client.bucket(bucket_name)

    file_name = filename


    blob = bucket.blob(file_name)



    ##Download the contents of the blob as a string
    content = blob.download_as_string()

    # Use BytesIO to convert the string to a file-like object

    file_obj = BytesIO(content)

    # # Initialize a client

    # storage_client = storage.Client()
    #
    # # Get the bucket and blob

    # bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(file_name)



    #Download the contents of the blob as a string
    content = blob.download_as_string()



    # Use BytesIO to convert the string to a file-like object

    file_obj = BytesIO(content)





    #Read the CSV file directly into a pandas DataFrame
    df_ = pd.read_csv(file_obj)

    return df_


# df = load_csv_from_gcs("notebot-backend","ai_engine_hackathon/dummy.csv")

# logging.warning(df.head())

# Add a new column with a constant value
df = pd.concat([df, df])

# This writes a pandas dataframe to a gcs bucket
def write_to_gcs(bucket_name,filename, data):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    csv = df.to_csv(index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv, content_type='text/csv')

    logging.warning("SUCCESS: Uploaded csv to GCP bucket")



# write_to_gcs("notebot-backend","ai_engine_hackathon/dummy.csv",df) 