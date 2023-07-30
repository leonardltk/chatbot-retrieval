import os
import shutil
import pprint
import pdb
import traceback
import pandas as pd
from termcolor import colored
import functools
import torch

import openai
import tiktoken

import langchain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

class HFInstructorEmbeddings():
    def __init__(self, 
                 embedding_model_name, 
                 CHUNKS_TXT, CHUNK_SIZE, CHUNK_OVERLAP
                 ) -> None:

        # global params
        self.CHUNKS_TXT = CHUNKS_TXT
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP

        # Model params
        self.embedding_model_name = embedding_model_name
        self.embedding_cost_per_token = 0
        self.model_kwargs = {'device': 'cpu'} # {"device": "cuda"}

        # to be initialised
        self.embedding = None
        self.vectordb = None

    def load_model(self):
        # Load model
        self.embedding = HuggingFaceInstructEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs,
            )

    # --- table helper functions
    def compute_table_cost(self,):
        return 0

        # Calculate cost to compute this table
        total_tokens = 0
        for sentence in self.vectordb.get()['documents']:
            encoding = tiktoken.encoding_for_model(self.embedding_model_name)
            token_lst = encoding.encode(sentence)
            total_tokens += len(token_lst)

        # Cost per token (in dollars)
        total_cost = total_tokens * self.embedding_cost_per_token
        formatted_cost = format(total_cost, '.10f').rstrip('0').rstrip('.')
        print(colored(f"total_tokens = {total_tokens}", 'blue'))
        # print(colored(f"The cost for making this table: ${total_cost}", 'blue'))
        print(colored(f"The cost for making this table: ${formatted_cost}", 'blue'))

        # Cost per token (in dollars)
        # table_cost_dict = {'Cost to make this table': total_cost}

        return total_cost

    # ---
    def write_to_chunks_txt(self, chunks_df):
        # Write to file <- chunks_df
        with open(self.CHUNKS_TXT, 'w', encoding='utf-8') as f:
            for page_content in chunks_df['page_content']:
                f.write(f"\n{page_content}\n")
            print(f'Finished writing to {self.CHUNKS_TXT}')

    def update_chunks_csv_df(self, chroma_dir):
        chunks_csv = os.path.join(chroma_dir, 'chunks.csv')
        # make chunks dict
        chunks_dict = {'ids':[], 'page_content':[], 'metadata':[]}
        for ids, page_content, metadata in zip(
            self.vectordb.get()['ids'], 
            self.vectordb.get()['documents'], 
            self.vectordb.get()['metadatas']
        ):
            chunks_dict['ids'].append(ids)
            chunks_dict['page_content'].append(page_content)
            chunks_dict['metadata'].append(metadata)
        # make chunks df
        chunks_df = pd.DataFrame(chunks_dict)
        # make chunks csv
        chunks_df.to_csv(chunks_csv, index=False)
        # update chunks.txt
        self.write_to_chunks_txt(chunks_df)


    # ---
    def make_vector_db(self, split_documents, chroma_dir, force):
        # load from disk
        if (not force) and os.path.exists(chroma_dir):
            print(f'make_vector_db: Table already created, skipping...')
            self.vectordb = Chroma(
                persist_directory=chroma_dir, 
                embedding_function=self.embedding
                )
            return

        # remove old database files if any
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)

        # Init
        os.makedirs(chroma_dir)
        chunks_csv = os.path.join(chroma_dir, 'chunks.csv')

        # store into csv
        if len(split_documents):
            chunks_df = pd.DataFrame({
                'page_content': [i.page_content for i in split_documents],
                'metadata': [i.metadata for i in split_documents],
            })
        else:
            chunks_df = pd.DataFrame({'page_content': [], 'metadata': []})
        chunks_df.to_csv(chunks_csv, index=False)

        # Write to file
        self.write_to_chunks_txt(chunks_df)

        # store the database
        print(f'make_vector_db: storing split documents into {chroma_dir}')
        vectordb = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embedding,
            persist_directory=chroma_dir
        )
        number_of_database_points = self.vectordb._collection.count()
        print(f'make_vector_db: number_of_database_points = {number_of_database_points}')

        return vectordb


    # ---
    def split_document(self, document_path, chunk_size, chunk_overlap,):
            
        # Initialise the loader
        if '.pdf' in document_path:
            loader = PyPDFLoader(document_path)
        else:
            loader = TextLoader(document_path)

        # Loading the document
        docs = loader.load()

        # Initialise the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

        # Perform the split
        document_lst = text_splitter.split_documents(docs) # list of Documents()

        return document_lst

    def document_to_vectordb(self, filename, chroma_dir_input, table_mode):
        print(colored('document_to_vectordb:', 'yellow', attrs=['bold']))

        if table_mode == 'create':
            chroma_dir = chroma_dir_input
        elif table_mode == 'add':
            chroma_dir = chroma_dir_input.rstrip('/') + '_new'

        # Split the document into chunks
        split_documents = self.split_document(filename, 
                                              chunk_size=self.CHUNK_SIZE, 
                                              chunk_overlap=self.CHUNK_OVERLAP,
                                              )

        # Post process
        for idx, chunk in enumerate(split_documents):
            chunk.page_content = chunk.page_content.replace('\n', ' ') 
            chunk.page_content = f"{idx}. {chunk.page_content}"

        # Store to table
        vectordb_to_add = self.make_vector_db(split_documents,
                                              chroma_dir,
                                              force=True)

        # Create or Add to current table
        if table_mode == 'create':
            self.vectordb = vectordb_to_add
        elif table_mode == 'add':
            # check for duplicates
            seen_documents = self.vectordb.get()['documents']
            for idx, documents in enumerate(vectordb_to_add.get()['documents']):
                if documents in seen_documents:
                    continue
                print(f'adding {idx} | {documents}')
                self.vectordb._collection.add(
                    ids=vectordb_to_add.get()['ids'][idx],
                    documents=vectordb_to_add.get()['documents'][idx],
                    metadatas=vectordb_to_add.get()['metadatas'][idx],
                )

            # update chunks.csv
            self.update_chunks_csv_df(chroma_dir_input)

        # status message
        status_msg = f'Completed generating {filename} to {chroma_dir}'

        # So next time we don't need to re-generate, just load
        self.vectordb.persist()

        return status_msg
