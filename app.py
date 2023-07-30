import os
import pdb
import traceback
import pandas as pd
from termcolor import colored
import functools
from dotenv import find_dotenv, load_dotenv

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import langchain

from embedding_helper import HFInstructorEmbeddings
from inference_helper import ChatModel, LlamaModel

# --- CONFIG
CHROMA_DIR = 'docs/chroma/'
UPLOAD_FOLDER = 'docs/uploads/'
CHUNKS_TXT = 'static/chunks.txt'
CHUNK_SIZE = 500 # 150000
CHUNK_OVERLAP = 200 # 0
RETRIEVER_KWARGS = {
        'search_type': 'similarity',
        'samples': 5
}

# --- Set up
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def read_env(env_path, api_version):
    env_dict = {}
    for line in open(env_path, 'r'):
        ENV_key, ENV_value = line.rstrip('\n').split('=')
        env_dict[ENV_key] = ENV_value
    return env_dict[api_version]


# --- Create table
@app.route('/upload_file', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'status': 'No file part in the request.'}), 400
    file = request.files['file']

    if file:
        # Check if the user has selected a file
        if file.filename == '':
            return jsonify({'status': 'No selected file.'}), 400

        # Check the file extension
        if not '.' in file.filename or \
            file.filename.rsplit('.', 1)[-1].lower() != 'pdf':
            return jsonify({'status': 'File extenstion should be .pdf'}), 400

        # init file path
        filename = secure_filename(file.filename)
        saved_filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save file locally
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(saved_filepath)

        # Process the file and store it to the table
        status = embedding_class.document_to_vectordb(saved_filepath, CHROMA_DIR, table_mode='create')

        # Compute table cost
        table_cost = embedding_class.compute_table_cost()

        return jsonify({'status': status, 'table_cost':table_cost})

    return jsonify({'status': 'File not allowed.'}), 400

# --- Add table
@app.route('/add_file', methods=['POST'])
def add_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'status': 'No file part in the request.'}), 400
    file = request.files['file']

    if file:
        # Check if the user has selected a file
        if file.filename == '':
            return jsonify({'status': 'No selected file.'}), 400

        # Check the file extension
        if not '.' in file.filename or \
            file.filename.rsplit('.', 1)[-1].lower() != 'pdf':
            return jsonify({'status': 'File extenstion should be .pdf'}), 400

        # init file path
        filename = secure_filename(file.filename)
        saved_filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save file locally
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(saved_filepath)

        # Process the file and store it to the table
        status = embedding_class.document_to_vectordb(saved_filepath, CHROMA_DIR, table_mode='add')

        # Compute table cost
        table_cost = embedding_class.compute_table_cost()

        return jsonify({'status': status, 'table_cost':table_cost})

    return jsonify({'status': 'File not allowed.'}), 400


# --- Chat
@app.route('/ask', methods=['POST'])
def ask():
    # Get the user's input from the request
    data = request.get_json()
    user_input = data['input']

    model_output, citations, inference_costs = chat_class.process_user_input(user_input,
                                                                             embedding_class.vectordb,
                                                                             RETRIEVER_KWARGS
                                                                             )

    # Return the model's output
    return jsonify({
         'output': model_output, 
         'citations': citations, 
         'embedding_cost': inference_costs['embedding_cost'], 
         'prompt_cost': inference_costs['prompt_cost'], })

# --- Summarise to Email
@app.route('/summarise', methods=['POST'])
def summarise():
    # Langchain inference
    model_output, citations, inference_costs = chat_class.perform_summary_and_email()

    # Return the model's output
    return jsonify({
         'output': model_output, 
         'citations': citations, 
         'embedding_cost': inference_costs['embedding_cost'], 
         'prompt_cost': inference_costs['prompt_cost'],
         })


# --- Start
if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # Init chatbot with conversation
    langchain.debug = True

    if 1: # Embedding model
        # HF Instructor Embeddings
        embedding_model_name = "hkunlp/instructor-xl"
        embedding_model_name = "hkunlp/instructor-base"
        embedding_class = HFInstructorEmbeddings(embedding_model_name,
                                                 CHUNKS_TXT, CHUNK_SIZE, CHUNK_OVERLAP,)
        embedding_class.load_model()
        embedding_class.make_vector_db(split_documents=[],
                                       chroma_dir=CHROMA_DIR, 
                                       force=False)

    if 0: # Inference model (google t5)
        model_name = "google/flan-t5-base"
        chat_class = ChatModel(model_name)
    if 1: # Inference model (llama2)
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        chat_class = LlamaModel(model_name, HF_ACCESS_TOKEN=os.getenv("HF_ACCESS_TOKEN"))

    # Start Flask app
    app.run(debug=False)
