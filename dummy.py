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
from pydantic import BaseModel
from typing import List, Dict

# Read file from GCS
#to upload the noisy hdbscan clusterer to GCP
from google.oauth2 import service_account

from google.cloud import storage
from datetime import datetime

# Replace with your OpenAI API key
openai_client = OpenAI(api_key="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")

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


# This writes a pandas dataframe to a gcs bucket
def write_to_gcs(bucket_name,filename, data):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    csv = data.to_csv(index=False)
    blob = bucket.blob(filename)
    blob.upload_from_string(csv, content_type='text/csv')

    logging.warning("SUCCESS: Uploaded csv to GCP bucket")

def generate_new_log(user_log, recent_student_performance):

    """
    Generate a new log given the old log and information about the students most recent performace

    Parameters:
    prompt (str): The prompt to send to GPT-4.

    Returns:
    str: The response from GPT-4.
    """

    chosen_model = "gpt-4o"
    # chosen_model = "gpt-4o-2024-08-06"
    # chosen_model = "gpt-4o-mini"
    
    # user_prompt, system_role_prompt = generate_combined_prompt(prompt)
    user_log = user_log # input from log 
    recent_student_performance = recent_student_performance # input from log 
    system_role_prompt =   f"""You are a part of a system that is designed to help students learn. 
    Your role is to:
    1. Document the strength and weaknesses of the student given their previous performance
    You are given:
    1. A previous log of student's performance, which describes their aptitude in a given subject
    2. The most recent performance of the student, which describes their learnings since the last log input

    Your task:
    Given these two inputs, generate a new log of the students performance. Match the previous tone, 
    style and brevity of the previousl log, whilst maintaining all relevat information.

    Here is the most recent entry from the log:
    {user_log} 

    Here is the current student performance:
    {recent_student_performance}
    """
    
    response = openai_client.chat.completions.create(
      model=chosen_model,
      messages=[
        {"role": "system", "content": f"""{system_role_prompt}"""},
      ]
    )
    return response.choices[0].message.content


class QuizOption(BaseModel):
    A1: str
    A2: str
    A3: str
    A4: str

class QuizQuestion(BaseModel):
    question: str
    options: QuizOption

class Quiz(BaseModel):
    questions: List[QuizQuestion]

def generate_quiz(user_log, topic):
    system_prompt = f"""Generate a 5-question multiple choice quiz to test a student's understanding of {topic}.
    Each question should have 4 options (A1, A2, A3, A4). 
    Focus on areas where the student needs improvement, use this as a guide: 
    {user_log}
    Format the response as a JSON object matching this schema:
    {{
        "questions": [
            {{
                "question": "question text",
                "options": {{
                    "A1": "first option",
                    "A2": "second option",
                    "A3": "third option",
                    "A4": "fourth option"
                }}
            }}
        ]
    }}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        response_format={ "type": "json_object" }
    )
    
    # Parse the response into our Pydantic model
    quiz_data = response.choices[0].message.content
    quiz = Quiz.model_validate_json(quiz_data)  # Updated from parse_raw
    return quiz


def mark_question(questions, answer,list_of_possible_answers):
    system_prompt = f"""You are a helpful assistant that marks multiple choice questions.
    You are given a question and the student's answer.
    You need to mark the question based what you know about the topic.

    Here is the question:
    {questions}

    Here is the student's answer:
    {answer}

    Here is the list of possible answers:
    {list_of_possible_answers}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
    )

    return response.choices[0].message.content

def generate_student_feedback(question_feedback_list):

    system_prompt = f"""You are a helpful assistant that generates feedback for a student based on their performance on a question.
    You are receiving the student's respnoses to a number of questions.

    You need to generate a summary of the student's performance.
    
    Highlight their strengths

    Highlight the areas where the student has demonstrated difficulty.

    And suggest areas for improvement.

    Here is what you need to review:
    {question_feedback_list}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
    )   

    return response.choices[0].message.content
    
    
    
    

# def get_tutor(user_prompt, ):

#     """
#     Send a prompt to the OpenAI API to get a response from GPT-4.

#     Parameters:
#     prompt (str): The prompt to send to GPT-4.

#     Returns:
#     str: The response from GPT-4.
#     """

#     chosen_model = "gpt-4o"
#     # chosen_model = "gpt-4o-2024-08-06"
#     # chosen_model = "gpt-4o-mini"
    
#     # user_prompt, system_role_prompt = generate_combined_prompt(prompt)
#     user_prompt = user_prompt# input from log 
#     system_role_prompt =   """You are a knowledgeable and patient tutor. Your role is to:
#     1. Help students understand concepts by breaking them down into simpler terms
#     2. Use analogies and examples to explain complex ideas
#     3. Ask guiding questions to help students arrive at answers themselves
#     4. Provide constructive feedback and encouragement
#     5. Identify and address any misconceptions in the student's understanding
#     6. Explain step-by-step solutions when needed
#     7. Maintain a supportive and engaging tone throughout the interaction

#     When responding to questions:
#     - First acknowledge the student's question
#     - Break down complex topics into digestible parts
#     - Use clear, simple language while maintaining accuracy
#     - Provide relevant examples or analogies
#     - Check for understanding and offer to clarify any points
#     - Encourage critical thinking and deeper exploration of the topic"""


#     # Load the tokenizer
#     tokenizer = tiktoken.get_encoding("cl100k_base")
    
#     def count_tokens(text):
#         tokens = tokenizer.encode(text)
#         return len(tokens)

#     num_tokens = count_tokens(system_role_prompt)
#     print(f"Number of system role tokens: {num_tokens}")
    
#     response = client.chat.completions.create(
#       model=chosen_model,
#       messages=[
#         {"role": "system", "content": f"""{system_role_prompt}"""},
#         {"role": "user", "content": user_prompt}
#       ]
#     )
#     return response.choices[0].message.content 

# user_prompt = "This course is first year undegrad statistics. This module is Bayesian probability. The student understands likelihood. This student has struggled with inference"




# # df = load_csv_from_gcs("notebot-backend","ai_engine_hackathon/dummy.csv")

# # logging.warning(df.head())

# # Add a new column with a constant value
# # df = pd.concat([df, df])



fake_student_performance = """The student has demonstrated difficulty with statistical inference concepts. 
Specifically, they:
- Struggle to understand the relationship between sample and population parameters
- Have trouble interpreting confidence intervals and their practical significance
- Show confusion when dealing with hypothesis testing procedures
- Need additional support in understanding p-values and their interpretation
- Can follow mechanical steps but lack deeper conceptual understanding
- Are proficient with likelihood calculations but struggle to connect this to inference
- Show hesitation when asked to draw conclusions from statistical analyses

However, they have shown:
- Good grasp of basic probability concepts

- Strong mathematical calculation abilities
- Eagerness to learn and willingness to ask questions
- Consistent completion of assigned practice problems"""

fake_user_log = """The student has demonstrated difficulty with statistical inference concepts. 
Specifically, they:
- Struggle to understand the relationship between sample and population parameters
- Have trouble interpreting confidence intervals and their practical significance
- Show confusion when dealing with hypothesis testing procedures
- Need additional support in understanding p-values and their interpretation
- Can follow mechanical steps but lack deeper conceptual understanding
- Are proficient with likelihood calculations but struggle to connect this to inference
- Show hesitation when asked to draw conclusions from statistical analyses

However, they have shown:
- Good grasp of basic probability concepts
- Strong mathematical calculation abilities
- Eagerness to learn and willingness to ask questions
- Consistent completion of assigned practice problems"""    

# test_user_log = generate_new_log(fake_user_log, fake_student_performance)
# logging.warning(test_user_log)

question_feedback = []

# Example usage:
quiz = generate_quiz(fake_user_log, "bayesian probability")
for i, q in enumerate(quiz.questions, 1):
    print(f"\nQuestion {i}: {q.question}")
    print(f"A1: {q.options.A1}")
    print(f"A2: {q.options.A2}")
    print(f"A3: {q.options.A3}")
    print(f"A4: {q.options.A4}")

    answer = input("Enter your answer (A1, A2, A3, A4): ")
    mark_response = mark_question(q.question, answer, [q.options.A1, q.options.A2, q.options.A3, q.options.A4])
    logging.warning(mark_response)

    question_feedback.append(mark_response)

student_feedback_report = generate_student_feedback(question_feedback)
logging.warning(student_feedback_report)

# Generate a new log
new_log = generate_new_log(fake_user_log, student_feedback_report)
logging.warning(new_log)

try:
    # download log from gcs
    log_file = load_csv_from_gcs("notebot-backend", "ai_engine_hackathon/dummy.csv")
    logging.warning(log_file)

    new_log_file = pd.DataFrame()

    # as a date yyyy_mm_dd
    new_log_file["last_worked_on"] = datetime.now().strftime("%Y_%m_%d")
    new_log_file["user_log"] = new_log
    new_log_file["topic"] = "Statistics"
    new_log_file["subject"] = "bayesian probability"

    # append the new log to the existing log
    log_file = pd.concat([log_file, new_log_file], ignore_index=True)

    # upload log to gcs
    write_to_gcs("notebot-backend", "ai_engine_hackathon/dummy.csv", log_file)

except Exception as e:
    logging.warning(f"Error downloading log file: {e}")

    new_log_file = pd.DataFrame()


# as a date yyyy_mm_dd
new_log_file["last_worked_on"] = datetime.now().strftime("%Y_%m_%d")
new_log_file["user_log"] = new_log
new_log_file["topic"] = "Statistics"
new_log_file["subject"] = "bayesian probability"

# upload log to gcs
write_to_gcs("notebot-backend", "ai_engine_hackathon/dummy.csv", new_log_file)
