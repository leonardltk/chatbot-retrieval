import os
import shutil
import pprint
import pdb
import traceback
import pandas as pd
from termcolor import colored
import functools
import torch
from torch import cuda, bfloat16

import openai
import tiktoken

import langchain
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import pipeline

def calculate_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    token_lst = encoding.encode(text)
    return len(token_lst)


class ChatModel():
    def __init__(self, 
                 model_name, 
                 ) -> None:

        # Model params
        self.model_name = model_name
        self.temperature = 0.5
        self.max_length = 1024
        self.top_p = 0.95
        self.repetition_penalty = 1.15

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        """
        saved to ~/.cache/huggingface/hub/$model_name/snapshots/RANDOM_HASH/spiece.model
        saved to ~/.cache/huggingface/hub/$model_name/snapshots/RANDOM_HASH/tokenizer.json
        saved to ~/.cache/huggingface/hub/$model_name/snapshots/RANDOM_HASH/tokenizer_config.json
        saved to ~/.cache/huggingface/hub/$model_name/snapshots/RANDOM_HASH/special_tokens_map.json
        """

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        """
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                    load_in_8bit=True,
                                                    device_map='auto',
                                                    # load_in_8bit_fp32_cpu_offload=True,
                                                    # torch_dtype=torch.float16,
                                                    # low_cpu_mem_usage=True,
                                                    )
        saved to ~/.cache/huggingface/hub/$model_name/snapshots/RANDOM_HASH/config.json
        saved to ~/.cache/huggingface/hub/$model_name/snapshots/RANDOM_HASH/model.safetensors
        """

        # initialise the pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model, 
            tokenizer=self.tokenizer, 
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)

        # chat history memory
        self.chat_history = []
        self.total_citation_dict = {}

    # ----- chat bot -----
    def get_standalone_question(self, chat_history, user_input):
        if len(chat_history):
            prompt = f"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {user_input}
Standalone question:"""
            standalone_question = self.local_llm(prompt)
        else:
            standalone_question = user_input

        return standalone_question

    def get_context(self, standalone_question, vector_db, vector_kwargs):
        search_type = vector_kwargs['search_type']
        samples = vector_kwargs['samples']

        # get documents
        if search_type == 'similarity':
            documents = vector_db.similarity_search(standalone_question, k=samples )
        elif search_type == 'similarity_score_threshold':
            score_threshold = vector_kwargs['score']
            context_with_score = vector_db.similarity_search_with_score(standalone_question, k=samples )
            documents = [i[0] for i in context_with_score if i[1]>=score_threshold]
        elif search_type == 'mmr':
            documents = vector_db.max_marginal_relevance_search(standalone_question, k=samples, fetch_k=samples*2)
        else:
            documents = []

        # get context as string
        context_lst = [d.page_content for d in documents]
        context_string = '\n\n'.join(context_lst)

        # get citations
        citation_dict = {}
        for i in documents:
            filename = i.metadata['source']
            page_number = i.metadata['page']
            # update locally
            if not filename in citation_dict:
                citation_dict[filename] = set()
            citation_dict[filename].add(page_number)
            # update globally
            if not filename in self.total_citation_dict:
                self.total_citation_dict[filename] = set()
            self.total_citation_dict[filename].add(page_number)
        # collate to list
        citations_lst = []
        for filename, page_numbers in citation_dict.items():
            citation = f"{filename} | pages={page_numbers}"
            citations_lst.append(citation)
        # condense to string
        citations = '\n'.join(citations_lst)

        return documents, context_string, citations

    def manual_qa(self, context, standalone_question):
        prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {standalone_question}
Helpful Answer:"""
        response_from_LLM = self.local_llm(prompt)
        return response_from_LLM

    def update_citation(self, result):
        citation_dict = {}
        for i in result['source_documents']:
            filename = i.metadata['source']
            page_number = i.metadata['page']
            # update locally
            if not filename in citation_dict:
                citation_dict[filename] = set()
            citation_dict[filename].add(page_number)
            # update globally
            if not filename in self.total_citation_dict:
                self.total_citation_dict[filename] = set()
            self.total_citation_dict[filename].add(page_number)

        # collate to list
        citations_lst = []
        for filename, page_numbers in citation_dict.items():
            citation = f"{filename} | pages={page_numbers}"
            citations_lst.append(citation)

        # condense to string
        citations = '\n'.join(citations_lst)

        return citations

    def process_user_input(self, user_input, vector_db, vector_kwargs):
        # Langchain inference 
        """
        # 0) Normal inference for first query from user
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer,
                just say that you don't know,
                don't try to make up an answer.
            <context 1>
            <context k>
            Question: <user query>
            Helpful Answer:
        # 1) Rephrase user's query into standalone question based on chat history context.
            Given the following conversation and a follow up question,
                rephrase the follow up question to be a standalone question,
                in its original language.
            Chat History:
                Human: <...>
                Assistant: <...>
                Human: <...>
                Assistant: <...>
                Human: <...>
                Assistant: <...>
            Follow Up Input: <user query>
            Standalone question:
        # 2) Inference using chromaDB context with this standalone question instead
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer,
                just say that you don't know,
                don't try to make up an answer.
            <context 1>
            <context k>
            Question: <standalone question>
            Helpful Answer:
        """

        print(colored(f'process_user_input({user_input})', 'red'))

        # Combine chat_history from list to string
        print(colored(f'\tchat_history = {self.chat_history}', 'red'))
        # self.chat_history = chat_history_dict['history']
        chat_history_str = '\n'.join([ f'Human: {human_response}\nAssistant: {agent_response}' for human_response, agent_response in self.chat_history])

        # Rephrase question
        standalone_question = self.get_standalone_question(chat_history_str, user_input)
        # Get context from ChromaDB
        documents, context, citations = self.get_context(standalone_question, vector_db, vector_kwargs)
        # Get answer from context
        response_from_LLM = self.manual_qa(context, standalone_question)

        # Update conversation
        self.chat_history.append((user_input, response_from_LLM))

        # token counting
        prompt_cost = 0
        # Calculate embedding costs
        embedding_cost = 0
        # inference_costs
        inference_costs = {
            'embedding_cost': embedding_cost,
            'prompt_cost': prompt_cost,
        }
        return response_from_LLM, citations, inference_costs

    # ----- summarising into email -----
    def perform_summary_and_email(self, ):
        print(colored(f'perform_summary_and_email()', 'red'))

        # Combine chat_history from list to string
        chat_history_str = '\n'.join([ f'Human: {human_response}\nAssistant: {agent_response}' for human_response, agent_response in self.chat_history])
        pprint.pprint(self.chat_history)
        print('chat_history_str', chat_history_str)

        # Create prompt
        prompt_chat_history = f"""Given the following chat history:
{chat_history_str}
Summarise the facts from the chat history, then present these summary into an email form.
Don't try to make up an answer.
The output should follow this format:
Subject: <Header of the summary>
Dear [Recipient],
<Insert summarisation here>
Thank you.
Regards,
[Relationship Manager]"""

        # Create prompt
        response_from_LLM = self.local_llm(prompt_chat_history)

        # citation
        citations_lst = []
        for filename, page_numbers in self.total_citation_dict.items():
            citation = f"{filename} | pages={page_numbers}"
            citations_lst.append(citation)
        citations = '\n'.join(citations_lst)

        # inference_costs
        inference_costs = {
            'embedding_cost': 0,
            'prompt_cost': 0,
        }

        return response_from_LLM, citations, inference_costs


class LlamaModel():

    def __init__(self, 
                 model_name, 
                 HF_ACCESS_TOKEN,
                 ) -> None:

        # --- Model params ---
        self.model_name = model_name
        self.temperature = 0.5
        self.pipeline_params = {
            'max_new_tokens': 512,
            'top_p': 0.95,
            'top_k': 30,
            'repetition_penalty': 1.15,
            'num_return_sequences': 1,
            'do_sample': True,
        }

        # --- Init params ---
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        self.email_template = f"""Subject: <Header of the summary>\n\nDear [Recipient],\n\n<Insert summarisation here>\n\nThank you.\n\nRegards,\n[Relationship Manager]"""


        # --- Load tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_auth_token=HF_ACCESS_TOKEN,)

        # --- Load model ---
        self.init_model(HF_ACCESS_TOKEN)


        # --- Initialise the pipeline ---
        self.pipe = pipeline("text-generation",
                        model=self.model, 
                        tokenizer=self.tokenizer, 
                        eos_token_id=self.tokenizer.eos_token_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        **self.pipeline_params
                    )
        self.local_llm = HuggingFacePipeline(pipeline=self.pipe,
                                             model_kwargs = {'temperature':self.temperature})

        # --- chat history memory ---
        self.chat_history = []
        self.total_citation_dict = {}


    # ----- load the model differently -----
    def init_model(self, HF_ACCESS_TOKEN):
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items
        self.model_config = transformers.AutoConfig.from_pretrained(
            self.model_name,
            use_auth_token=HF_ACCESS_TOKEN
        )

        # load model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            config=self.model_config,
            quantization_config=self.bnb_config,
            device_map='auto',
            use_auth_token=HF_ACCESS_TOKEN
        )

        self.model.eval()

        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        print(f"Model loaded on {device}")


    # ----- chat bot -----
    # ----- Make question with history -----
    def setup_prompt_with_tokens(self, chat_history, user_input, system_prompt=None):
        # Init system prompt with tokens
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        system_prompt_token = f"{self.B_SYS}{system_prompt}{self.E_SYS}"

        # chat_history is empty
        if len(chat_history) == 0:
            prompt_str = f"{self.B_INST} {system_prompt_token}{user_input} {self.E_INST}"
            return prompt_str

        # chat_history is not empty,
        # iteratively append to list with tokens
        prompt_lst = []
        for idx, (prev_user_req, prev_asst_res) in enumerate(chat_history):
            if idx == 0:
                prompt_lst.append(f"{self.B_INST} {system_prompt_token}{prev_user_req} {self.E_INST} {prev_asst_res} ")
            else:
                prompt_lst.append(f"{self.B_INST} {prev_user_req} {self.E_INST} {prev_asst_res} ")

        # Add new user request
        prompt_lst.append(f"{self.B_INST} {user_input} {self.E_INST}")

        # Merge to string
        prompt_str = '\n'.join(prompt_lst)
        return prompt_str

    def get_standalone_question(self, chat_history, user_input):
        # no history
        if len(chat_history) == 0:
            return user_input

        system_prompt = """You are a helpful assistant in rephrasing question using the memory of chat history.
Given a chat history, and a new question, provide a relevant rephrased question.
Make sure the generated question is asking the same thing as the original question.
If there are no chat history, just repeat the question back.
You should not refuse to answer questions.
Don't ever thank the user.
Don't provide simulation or fictional scenerio, just rephrase the question.
If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you.
Don't provide info you weren't asked to provide."""

        prompt = self.setup_prompt_with_tokens(chat_history, user_input, system_prompt)
        prompt += " Rephrased relevant question: "
        print(colored(prompt, 'red'))

        standalone_question = self.local_llm(prompt)
        standalone_question = standalone_question.strip('\n ')
        print(colored(standalone_question, 'blue'))

        return standalone_question

    # ----- Make question with history -----
    def get_prompt(self, instruction, new_system_prompt=None ):
        if new_system_prompt is None:
            new_system_prompt = self.DEFAULT_SYSTEM_PROMPT
        SYSTEM_PROMPT = self.B_SYS + new_system_prompt + self.E_SYS
        prompt_template =  self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def manual_qa(self, context, rephrased_qns, raw_qns):
        system_prompt = """You are a very helpful assistant.
Use the only the information from the context to answer the question at the end.
Do not use any information other than the provided context.
If you dont know the answer, just say i don't know.
Find the closest answer from the context.
Always answer as helpfully as possible for the user.
You should not refuse to answer questions.
Don't correct the user.
Don't ever thank the user.
Don't engage in conversation.
If asked for an opinion express one!!
If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. 
Don't provide info you weren't asked to provide.
Just provide the answer, do not use long sentences."""

        standalone_question = f"Questions: {raw_qns} {rephrased_qns}"
        instruction = f"""Context:\n\n{context}\n\n{standalone_question}"""

        prompt = self.get_prompt(instruction, system_prompt)
        prompt += " Relevant answer from the context:"
        print(colored(prompt, 'red'))

        response_from_LLM = self.local_llm(prompt);             print(colored(response_from_LLM, 'blue'))

        return response_from_LLM

    # ----- context/citation -----
    def get_context(self, standalone_question, vector_db, vector_kwargs):
        search_type = vector_kwargs['search_type']
        samples = vector_kwargs['samples']

        # get documents
        if search_type == 'similarity':
            documents = vector_db.similarity_search(standalone_question, k=samples )
        elif search_type == 'similarity_score_threshold':
            score_threshold = vector_kwargs['score']
            context_with_score = vector_db.similarity_search_with_score(standalone_question, k=samples )
            documents = [i[0] for i in context_with_score if i[1]>=score_threshold]
        elif search_type == 'mmr':
            documents = vector_db.max_marginal_relevance_search(standalone_question, k=samples, fetch_k=samples*2)
        else:
            documents = []

        # get context as string
        context_lst = [d.page_content for d in documents]
        context_string = '\n\n'.join(context_lst)

        # get citations
        citation_dict = {}
        for i in documents:
            filename = i.metadata['source']
            page_number = i.metadata['page']
            # update locally
            if not filename in citation_dict:
                citation_dict[filename] = set()
            citation_dict[filename].add(page_number)
            # update globally
            if not filename in self.total_citation_dict:
                self.total_citation_dict[filename] = set()
            self.total_citation_dict[filename].add(page_number)
        # collate to list
        citations_lst = []
        for filename, page_numbers in citation_dict.items():
            citation = f"{filename} | pages={page_numbers}"
            citations_lst.append(citation)
        # condense to string
        citations = '\n'.join(citations_lst)

        return documents, context_string, citations

    def update_citation(self, result):
        citation_dict = {}
        for i in result['source_documents']:
            filename = i.metadata['source']
            page_number = i.metadata['page']
            # update locally
            if not filename in citation_dict:
                citation_dict[filename] = set()
            citation_dict[filename].add(page_number)
            # update globally
            if not filename in self.total_citation_dict:
                self.total_citation_dict[filename] = set()
            self.total_citation_dict[filename].add(page_number)

        # collate to list
        citations_lst = []
        for filename, page_numbers in citation_dict.items():
            citation = f"{filename} | pages={page_numbers}"
            citations_lst.append(citation)

        # condense to string
        citations = '\n'.join(citations_lst)

        return citations

    # ----- chat bot -----
    def process_user_input(self, user_input, vector_db, vector_kwargs):
        # Langchain inference 
        """
        # 0) Normal inference for first query from user
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer,
                just say that you don't know,
                don't try to make up an answer.
            <context 1>
            <context k>
            Question: <user query>
            Helpful Answer:
        # 1) Rephrase user's query into standalone question based on chat history context.
            Given the following conversation and a follow up question,
                rephrase the follow up question to be a standalone question,
                in its original language.
            Chat History:
                Human: <...>
                Assistant: <...>
                Human: <...>
                Assistant: <...>
                Human: <...>
                Assistant: <...>
            Follow Up Input: <user query>
            Standalone question:
        # 2) Inference using chromaDB context with this standalone question instead
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer,
                just say that you don't know,
                don't try to make up an answer.
            <context 1>
            <context k>
            Question: <standalone question>
            Helpful Answer:
        """

        print(colored(f'process_user_input({user_input})', 'red'))

        # Rephrase question
        rephrased_question = self.get_standalone_question(self.chat_history, user_input)
        # Get context from ChromaDB
        documents, context, citations = self.get_context(rephrased_question, vector_db, vector_kwargs)
        # Get answer from context
        response_from_LLM = self.manual_qa(context, rephrased_question, user_input)

        # Update conversation
        self.chat_history.append((user_input, response_from_LLM))

        # token counting
        prompt_cost = 0
        # Calculate embedding costs
        embedding_cost = 0
        # inference_costs
        inference_costs = {
            'embedding_cost': embedding_cost,
            'prompt_cost': prompt_cost,
        }

        return response_from_LLM, citations, inference_costs


    # ----- summarising into email -----
    def manual_summary(self):
        # Create prompt
        # no history
        if len(self.chat_history) == 0:
            return self.email_template

        system_prompt = f"""You are a very helpful email generator assistant. \nSummarise the chat history, and follow the specified email format to draft an email containing the summarised information.\nKeep all the facts in the chat history.\nDon't provide info that were not in the chat history.\nDon't ever thank the user."""

        chat_history_str = '\n'.join([ f'Human: {human_response}\nAssistant: {agent_response}' for human_response, agent_response in self.chat_history])
        email_request = f"""Output the email content in the following email format delimited by triple backticks\n```\n{self.email_template}\n```"""
        user_prompt = f"""#####\nChat history\n#####\n{chat_history_str}\n\n#####\nEmail format\n#####\n{email_request}"""

        prompt = self.get_prompt(user_prompt, system_prompt)
        prompt += 'Email output:'
        print(colored(prompt, 'red'))

        response_from_LLM = self.local_llm(prompt)
        print(colored(response_from_LLM, 'blue'))

        return response_from_LLM

    def perform_summary_and_email(self, ):
        print(colored(f'perform_summary_and_email()', 'red'))

        # Create prompt
        response_from_LLM = self.manual_summary()
        chat_history_str = '\n'.join([ f'Human: {human_response}\nAssistant: {agent_response}' for human_response, agent_response in self.chat_history])
        print('chat_history_str', chat_history_str)

        # Create prompt
        prompt_chat_history = f"""Given the following chat history:
{chat_history_str}

Summarise the facts from the chat history, then present these summary into an email form.
Don't try to make up an answer.
The output should follow this format:

Subject: <Header of the summary>

Dear [Recipient],

<Insert summarisation here>

Thank you.

Regards,
[Relationship Manager]"""

        # Create prompt
        response_from_LLM = self.local_llm(prompt_chat_history)

        # citation
        citations_lst = []
        for filename, page_numbers in self.total_citation_dict.items():
            citation = f"{filename} | pages={page_numbers}"
            citations_lst.append(citation)
        citations = '\n'.join(citations_lst)

        # inference_costs
        inference_costs = {
            'embedding_cost': 0,
            'prompt_cost': 0,
        }

        return response_from_LLM, citations, inference_costs

# 
