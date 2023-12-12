from flask import Flask, jsonify, request, Response
import requests
import yaml
import os
import re
import json
import numpy as np
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
from typing import Literal, Optional, Any

app = Flask(__name__)

model_id = "meta-llama/llama-2-70b-chat"
# ... (import other libraries and functions)

class MiniLML6V2EmbeddingFunction():
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()

emb_function = MiniLML6V2EmbeddingFunction()


# Define a custom constructor for !Ref and !GetAtt tags
def ref_getatt_constructor(loader, node):
    if isinstance(node, yaml.ScalarNode):
        # Handle scalar values (non-sequence)
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        # Handle sequences (lists)
        return loader.construct_sequence(node)
    else:
        # Handle other cases
        raise ValueError(f"Unexpected node type: {node.tag}")
    
# Add the custom constructor to the yaml loader
yaml.add_constructor('!Ref', ref_getatt_constructor)

yaml.add_constructor('!GetAtt', ref_getatt_constructor)

yaml.add_constructor('!Join', ref_getatt_constructor)

yaml.add_constructor('!Sub', ref_getatt_constructor)

yaml.add_constructor('!Select', ref_getatt_constructor)

yaml.add_constructor('!FindInMap', ref_getatt_constructor)

yaml.add_constructor('!Or', ref_getatt_constructor)

yaml.add_constructor('!Equals', ref_getatt_constructor)

yaml.add_constructor('!If', ref_getatt_constructor)

def pdf_to_text(path: str,
                start_page: int = 1,
                end_page: Optional[int | None] = None) -> list[str]:
  """
  Converts PDF to plain text.

  Args:
      path (str): Path to the PDF file.
      start_page (int): Page to start getting text from.
      end_page (int): Last page to get text from.
  """
  loader = PyPDFLoader(path)
  pages = loader.load()
  total_pages = len(pages)

  if end_page is None:
      end_page = len(pages)

  text_list = []
  for i in range(start_page-1, end_page):
      text = pages[i].page_content
      text = text.replace('\n', ' ')
      text = re.sub(r'\s+', ' ', text)
      text_list.append(text)

  return text_list

def text_to_chunks(texts: list[str],
                   word_length: int = 150,
                   start_page: int = 1) -> list[list[str]]:
  """
  Splits the text into equally distributed chunks.

  Args:
      texts (str): List of texts to be converted into chunks.
      word_length (int): Maximum number of words in each chunk.
      start_page (int): Starting page number for the chunks.
  """
  text_toks = [t.split(' ') for t in texts]
  chunks = []

  for idx, words in enumerate(text_toks):
      for i in range(0, len(words), word_length):
          chunk = words[i:i+word_length]
          if (i+word_length) > len(words) and (len(chunk) < word_length) and (
              len(text_toks) != (idx+1)):
              text_toks[idx+1] = chunk + text_toks[idx+1]
              continue
          chunk = ' '.join(chunk).strip()
          chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
          chunks.append(chunk)

  return chunks

def get_text_embedding(texts: list[list[str]],
                       batch: int = 1000) -> list[Any]:
  """
  Get the embeddings from the text.

  Args:
      texts (list(str)): List of chucks of text.
      batch (int): Batch size.
  """
  embeddings = []
  for i in range(0, len(texts), batch):
      text_batch = texts[i:(i+batch)]
      # Embeddings model
      emb_batch = emb_function(text_batch)
      embeddings.append(emb_batch)
  embeddings = np.vstack(embeddings)
  return embeddings

def get_search_results(question, embeddings, chunks):
  """
  Get best search results
  """
  emb_question = emb_function([question])
  nn = NearestNeighbors(n_neighbors=4)
  nn.fit(embeddings)
  neighbors = nn.kneighbors(emb_question, return_distance=False)
  topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]

  return topn_chunks

def build_prompt(question, topn_chunks_for_prompts):

  '''
  build prompt for general Q&A
  '''

  prompt = ""
  prompt += 'Search results:\n'

  for c in topn_chunks_for_prompts:
      prompt += c + '\n\n'

  prompt += "Instructions: You are a cyber security expert. Carefully compose a concise reply to the query using the cyber security requirements in the search results given. "\
          "First answer yes or no before explaining. "\
          "Cite each reference using [Page Number] notation (every result has this number at the beginning). "\
          "Citation should be done at the end of each sentence. Only include information found in the results and "\
          "don't add any additional information. Make sure the answer is correct and don't output false content. "\
          "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
          "search results which has nothing to do with the question. Only answer what is asked. The "\
          "answer should be short and concise. Give any explanation in point form."

#   prompt += "Instructions: You are a cyber security expert. Carefully compose a concise reply to the query using the cyber security requirements in the search results given. "\
#           "First answer yes or no before explaining. "\
#           "Cite each reference using [Page Number] notation (every result has this number at the beginning). "\
#           "Citation should be done at the end of each sentence. Only include information found in the results and "\
#           "Give any explanation in point form."


  prompt += f"\n\n\nInput: {question}\n\nOutput: "

  return prompt

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    policy_file_there = 0
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    if 'policy_file' in request.files:
        policy_file_there = 1
        uploaded_policy_file = request.files.get("policy_file")

        policy_file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_policy_file.filename)
        uploaded_policy_file.save(policy_file_path)

            # Process the uploaded files
        text_list = pdf_to_text(policy_file_path)
        chunks = text_to_chunks(text_list)
        embeddings = get_text_embedding(chunks)

    uploaded_files = request.files.getlist("file")

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Save uploaded files to the upload folder
    file_paths = []

    list_rows = []
    df_results = pd.DataFrame(columns=["filename", "policy_file", "compliance", "reasons"])

    for file in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_paths.append(file_path)

        with open(file_path, 'r', encoding='utf-8') as this_file:
            yaml_content = yaml.load(this_file.read(), Loader=yaml.FullLoader)
        print(yaml_content)
        
        output_policy = ""
        output_filename = file.filename
        output_section = ""
        output_compliance = ""
        output_reasons = ""

        if yaml_content:

            output_section = yaml_content["Resources"]

            if policy_file_there == 1:
                output_policy = uploaded_policy_file.filename
                print(output_policy)

                prompt = "Input: You are an AWS cloud expert. What does this cloudformation template do? Answer concisely."
                prompt += "\n"
                prompt += str(yaml_content["Resources"])
                
                prompt += "\n"
                prompt += "Output:"

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + ibm_api_key
                }

                payload = {
                    "model_id": model_id,
                    "inputs": [prompt],
                    "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
                }
            
                response = requests.request("POST", ibm_cloud_url, json=payload, headers=headers)
                response_json = response.json()
                output = response_json.get("results")[0]["generated_text"]
                print(output)

                #Q&A functionalities
                question = output + "Is this compliant?"
                topn_chunks = get_search_results(question, embeddings, chunks)
                question = "Is the cloud formation template below compliant?:" + str(yaml_content["Resources"])
                prompt = build_prompt(question, topn_chunks)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + ibm_api_key
                }

                payload = {
                    "model_id": model_id,
                    "inputs": [prompt],
                    "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
                }
                response = requests.request("POST", ibm_cloud_url, json=payload, headers=headers)
                response_json = response.json()
                output = response_json.get("results")[0]["generated_text"]
                print(output)

                if "yes" in output[:5].lower():
                    output_compliance = "Yes"
                if "no" in output[:5].lower():
                    output_compliance = "No"

                output_reasons = output
                new_row = {"filename":output_filename, "policy_file":output_policy, "compliance":output_compliance, "reasons":output_reasons}
                list_rows.append(pd.DataFrame([new_row]))

            else:
                prompt = "Input: You are a cyber security expert. Is the cloud formation template below compliant according to AWS Foundations Benchmark v1.2.0?  "\
                    "First answer yes or no before explaining."\
                "Carefully compose a concise reply. "\
                "Make sure the answer is correct and don't output false content."\
                "answer should be short and concise. Give any explanation in point form."

                prompt += "\n"
                prompt += str(yaml_content["Resources"])
                prompt += "Output:"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + ibm_api_key
                }

                payload = {
                    "model_id": model_id,
                    "inputs": [prompt],
                    "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
                }
                response = requests.request("POST", ibm_cloud_url, json=payload, headers=headers)
                response_json = response.json()
                output = response_json.get("results")[0]["generated_text"]

                output_reasons = output

                if "yes" in output[:5].lower():
                    output_compliance = "Yes"
                if "no" in output[:5].lower():
                    output_compliance = "No"

                new_row = {"filename":output_filename, "policy_file":output_policy, "compliance":output_compliance, "reasons":output_reasons}
                list_rows.append(pd.DataFrame([new_row]))

        df_results = pd.concat(list_rows, ignore_index=True)

    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv')
    df_results.to_csv(csv_file_path, index=False)

    return Response(
        csv_file_path,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=result.csv"}
    )

if __name__ == '__main__':
    load_dotenv()
    ibm_api_key = os.getenv("IBM_API_KEY", None)
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)

    app.run(debug=True)
