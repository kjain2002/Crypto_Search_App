import weaviate
import weaviate
import json
# from agent import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.agents import Tool
from langchain import PromptTemplate
import ast
from pycoingecko import CoinGeckoAPI
from langchain.llms.openai import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
import os
import numpy as np
import openai
import datetime as dt
from datetime import timedelta
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain import LLMChain, PromptTemplate, OpenAI
from openai.embeddings_utils import cosine_similarity
from langchain.tools import BaseTool
import numpy as np
import pandas as pd
import os
import ast
import json
import requests
import time
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor, load_tools
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
llm = OpenAI(
    temperature=0,
    model_name='text-davinci-003'
)
df = pd.read_csv('df.csv')
df['embedding'] = df['call descriptions'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

def top_dics(list_dics, key, query, num):
    l = []
    for dic in list_dics:
        l.append(dic[key])
    nl = []
    if len(l) > 143:
        input_text = "Here is a list of " + key + ":" + str(l[:143]) + ". Filter this list to return a MAX OF " + str(num) + " relevant values based on following user input: '" + query + "'. Make sure to only return values from specified list. Return python list without any additional explanation, so I can directly do a literal eval on the output. Do not return empty lists!"
        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= input_text,
        max_tokens=len(input_text)
        )
        nl = ast.literal_eval(response.choices[0]['text'])
        if len(nl) < num:
            input_text = "Here is a list of " + key + ":" + str(l[143:]) + ". Filter this list to return a MAX OF " + str(num - len(nl)) + " relevant values based on following user input: '" + query + "'. Make sure to only return values from specified list. Return python list without any additional explanation, so I can directly do a literal eval on the output. Do not return empty lists!"
            response = openai.Completion.create(
            engine="text-davinci-003",
            prompt= input_text,
            max_tokens=len(input_text)
            )
            nl.extend(ast.literal_eval(response.choices[0]['text']))
    else:
        input_text = "Here is a list of " + key + ":" + str(l) + ". Filter this list to return a MAX OF " + str(num) + " relevant values based on following user input: '" + query + "'. Make sure to only return values from specified list. Return python list without any additional explanation, so I can directly do a literal eval on the output. Do not return empty lists!"
        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= input_text,
        max_tokens=len(input_text)
        )
        nl = ast.literal_eval(response.choices[0]['text'])    
    filtered_dics = []
    for dic in list_dics:
        if dic[key] in nl:
            filtered_dics.append(dic)
    return filtered_dics

def top_keys(keys, query, num):
    input_text = "Here is a list of keys in a dictionary: " + str(keys) + ". From this list only, return a python list of only top " + str(min(num, len(keys))) + " keys including 'id' that are relevant based on following user input: '" + query + "'. Return python list without any additional explanation, so I can directly do a literal eval on the output. Do not return empty lists!"

    # Call the OpenAI API
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt= input_text,
    max_tokens=len(input_text)
    )
    
    # Extract the generated text from the API response
    st = ast.literal_eval(response.choices[0]['text'].lower())
    return st

def reduce_dict_to_x(dictionary, query, x):
    # Otherwise, we need to call top_keys to find the top x keys
    keys = list(dictionary.keys())
    top_x_keys = top_keys(keys, query, x)
    if (len(keys) > x) and (len(top_x_keys) != x):
        print("something is wrong")
    
    # Create a new dictionary with only the top x keys and their values
    new_dict = {}
    for key in top_x_keys:
        if key in dictionary:
            new_dict[key] = dictionary[key]
    
    # Recursively reduce the nested dictionaries
    for key, value in new_dict.items():
        if isinstance(value, dict):
            new_dict[key] = reduce_dict_to_x(value, query, x)
    
    return new_dict

def all_purpose_defi(func_call_query):
    json_output = ""
    try:
        func_call_query = eval(func_call_query)
        json_output = json.loads(requests.get(func_call_query[0]).text)
        query = func_call_query[1]
        num = func_call_query[2]

        if type(json_output) == list and len(json_output) > 500:
            top_dics(json_output, top_keys(list(json_output.keys())), query, 100)

        if num != None:
            if (type(json_output) == list):
                if (type(json_output[0]) == dict):
                    final = []

                    timer_start = time.time()
                    for i in json_output:
                        if time.time() - timer_start > 20:  # n second timeout
                            return "Try Vector Search"
                        
                        new_dict = reduce_dict_to_x(i, query, num)
                        final.append(new_dict)
                    json_output = final
            elif type(json_output) == dict:
                if len(json_output) > 1:
                    new_dict = reduce_dict_to_x(json_output, query, num)
                    json_output = new_dict
                    
        res_length = len(str(json_output))/4
        if (json_output == []) or (json_output == None):
            json_output = "Try Google Search"
        elif res_length > 3800:
            json_output = "Try Google Search"
    except Exception as e:
        print("An error occurred:", e)
        json_output = "Try Google Search"
    return json_output

def construct_call(api_call, query): 
    template = """API Call: {api_call}
    Replace the word inside the curly braces with a keyword from the user input: "{query}", that matches it best.
    
    If the word in the curly braces the word "coins" in it, follow the below conventions:

    Sample input: What is the price of ethereum?
    Sample keywords: ethereum
    Sample API Call construction: https://coins.llama.fi/prices/current/coins -> https://coins.llama.fi/prices/current/coingecko:ethereum

    Sample input: What are the prices of ethereum and bitcoin?
    Sample keywords: ethereum, bitcoin
    Sample API Call construction: https://coins.llama.fi/prices/current/coins -> https://coins.llama.fi/prices/current/coingecko:ethereum,coingecko:bitcoin

    If it does not have the word "coins" in it, ignore the above conventions.

    Answer with just the completed call: """

    prompt = PromptTemplate(
        template=template, 
        input_variables=["api_call", "query"]
    )

    create_call_chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=True
    )
    
    return create_call_chain.run(api_call=api_call, query=query)

def vector_search(query):
    query_output = eval(query)
    query_new = query_output[0]
    index = query_output[1]
# connect to your url
    client = weaviate.Client(
        url = "https://t56fcskxqxwq2ghnixwzkg.gcp.weaviate.cloud",
        auth_client_secret=weaviate.AuthClientPassword(
                            username = "yield.disruption@gmail.com",
                            password = "BigGreenHatsAre@Home612",
                        ),
        additional_headers = {
            "X-OpenAI-Api-Key": "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"  # Replace with your API key
        }
    )
    print(type(index))
    nearText = {"concepts": [query_new]}
    result = (
        client.query
        .get("Question", ["text", "url"])
        .with_near_text(nearText)
        .with_limit(3)
        .do()
    )
    return json.dumps(result["data"]["Get"]["Question"][index], indent=4)

class AllPurposeTool(BaseTool):
    name = "AllPurposeTool"
    description = """
    Useful for when you need to access real time crypto data from the DefiLlama API.
    Pass in the user's question as the query in all lowercase."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        search_term_vector = get_embedding(query, engine='text-embedding-ada-002')
        df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
        top_calls = np.array(df.sort_values('similarities', ascending=False).head(1)['call prefixes'])[0]

        print(top_calls)

        if '{' in top_calls:
            complete_call = construct_call(top_calls, query)
            call_test = requests.get(complete_call)
        else:
            complete_call = top_calls
            call_test = requests.get(complete_call)

        if call_test.status_code == 200:
            final_call = complete_call
            return all_purpose_defi(repr([final_call, query, 2]))

        return "Try Vector Search"
        
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Does not support async")

class BridgeTool(BaseTool):
    name = "Endpoint: /bridge/id"
    description = """Pass in the user's question along with the name of the bridge as the query in all lowercase.

                     Sample input: "What is the weekly volume of Polygon?"
                     Sample query: "weekly volume of polygon"
    """

    def _run(self, query: str) -> str:
        """Use the tool."""
        bridges = json.loads(requests.get('https://bridges.llama.fi/bridges?includeChains=false').text)['bridges']
        for bridge in bridges:
            if bridge['name'] in query:
                endpoint = 'https://bridges.llama.fi/bridge/' + str(bridge['id'])
                return all_purpose_defi(repr([endpoint, query, None]))
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Does not support async")

tools = [
    AllPurposeTool(),
    BridgeTool(),
    Tool(
        name="vector_search best",
        func=vector_search,
        description = '''Put in call in this format: [original query, 0]
                         
                         Sample user input -> How is time in proof of stake divided in Ethereum?
                         Sample Action input -> ['How is time in proof of stake divided in Ethereum? (Include links in observation).', 0]'''
    ),
    Tool(
        name="vector_search second best",
        func=vector_search,
        description = '''Put in call in this format: [original query, 1]
                         
                         Sample user input -> How is time in proof of stake divided in Ethereum?
                         Sample Action input -> ['How is time in proof of stake divided in Ethereum? (Include links in observation)', 1]'''
    ),
    Tool(
        name="vector_search worst",
        func=vector_search,
        description = '''Put in call in this format: [original query, 2]
                         
                         Sample user input -> How is time in proof of stake divided in Ethereum?
                         Sample Action input -> ['How is time in proof of stake divided in Ethereum? (Include links in observation)', 2]'''
    ),
]

prefix = """You are an Artificial Intelligence with the purpose of answering questions regarding web3 topics. You have access to the following tools:"""
suffix = """Begin!"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

tool_names = [tool.name for tool in tools]

agent = ZeroShotAgent(
    llm_chain=llm_chain, 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True
)

# agent_executor.run('What is a DAO')

st.title("Defi Llama API Integration")
st.caption('_:blue[This integration supports questions about TVL, coins, stablecoins, yields, abi-decoder, bridges, volumes, and fees and revenue\n]_', 
unsafe_allow_html=True)
user_prompt = st.text_input("Enter a question. Do not use contractions!", "")
submit = st.button("Submit")
if submit:
    st.write(agent_executor.run(user_prompt))
