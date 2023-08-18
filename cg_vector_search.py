# IMPORTS
import weaviate
import json
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.agents import Tool
import ast
from pycoingecko import CoinGeckoAPI
from langchain.llms.openai import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain import LLMChain, OpenAI
from openai.embeddings_utils import cosine_similarity
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, Tool, AgentExecutor, load_tools
from openai.embeddings_utils import get_embedding
import pandas as pd
import warnings
from typing import Callable
import streamlit as st
import datetime as dt
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain import PromptTemplate
import numpy as np
from datetime import timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain import LLMChain, PromptTemplate, OpenAI
from openai.embeddings_utils import cosine_similarity
from langchain.tools import BaseTool
import time

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    
# os.environ['OPENAI_API_KEY'] = "..."
llm = OpenAI(temperature=0)
os.environ["SERPER_API_KEY"] = "d4247b1d0eadad3649101d95e6185c569b02bf19"
openai.api_key = "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"

search = GoogleSerperAPIWrapper()
cg = CoinGeckoAPI()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# GENERIC TOKEN REDUCTION FUNCTIONS (for API endpoint responses)


# VERTICAL REDUCTION: Finding top dictionaries of relevance in a list of dictionaries using openai 
# Use case: Most keys in each dictionary are relevant; however all dictionaries are not relevant

# list_dics = list of dictionaries, key = a key in the dictionaries used to determine relevance of a dictionary overall, query = What the user typed in search engine, num = number of dictionaries you want to retain
def top_dics(list_dics, key, query, num):
    # Iterating through all dictionaries and retaining the value of 'key' and putting them in a list: 'l'
    l = []
    for dic in list_dics:
        l.append(dic[key])

    # Iterating through 'l' and determining relevant values for 'key' and having openai return it as a string list that can be literal evaled: 'nl'
    nl = []

    # Using openai programatically also has a token limit, so if 'l' is larger than 143, have openai find relevant values in the first 143 values. If openai finds less than 'num' relevant values, have it search the rest of the list
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
    
    # Find dictionaries that correspond to the filtered values returned by openai
    filtered_dics = []
    for dic in list_dics:
        if dic[key] in nl:
            filtered_dics.append(dic)
    return filtered_dics




# HORIZONTAL REDUCTION: Finding top keys in a dictionary using openai 
# Use case: There is only one or few dictionaries, but with many keys or many nested levels. We need to reduce the size of these dictionaries

# keys = list of keys from dictionary, query = What the user typed in search engine, num = number of dictionaries you want to retain
def top_keys(keys, query, num):
    #Openai finding list of relevant keys based on user request
    input_text = "Here is a list of keys in a dictionary: " + str(keys) + ". From this list only, return a python list of only top " + str(min(num, len(keys))) + " keys including 'id' that are relevant based on following user input: '" + query + "'. Return python list without any additional explanation, so I can directly do a literal eval on the output. Do not return empty lists!"

    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt= input_text,
    max_tokens=len(input_text)
    )
    
    # Extract the generated text from the API response and literal eval it
    filtered_keys = ast.literal_eval(response.choices[0]['text'].lower())
    return filtered_keys

# Recursive function for implementing top_keys across all nested levels of a dictionary
# dictionary = dictionary that needs to be reduced, query = What the user typed in search engine, x = number of keys you want per nested level of dictionary
def reduce_dict_to_x(dictionary, query, x):
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TOKEN REDUCTION FUNCTIONS FOR SPECIFIC ENDPOINTS & TOOLS
    # Note: This was before I had created my generic token reduction tools. I did not have time to modify 
    # already existing tools to use generic functions due to limited time and other pressing priorities. Not
    # only would I have to spend time changing prompt engineering, I would also have to spend time testing.


# Filters get_coins_markets output to return market data for coins that have particular market cap ranks
def get_market_data_by_rank(currency, ranks):
    result = []
    dic = cg.get_coins_markets(currency)
    for coin in dic:
            if coin['market_cap_rank'] in ranks:
                result.append(coin)
    return result

# The entire coins list (and their details) always surpass token limits for langchain/openai, so this function filters for coins whose names/symbols began with a certain letter
# label = a key (usually name or symbol) denoting a coin, starts_with = a starting letter for a name
def get_coins_list(label = "", starts_with = ""):
    values_list = {}
    for dict_item in cg.get_coins_list():
        if (label in dict_item) and (dict_item[label].lower().startswith(starts_with.lower())):
            values_list[dict_item['id']]= dict_item[label]
    if len(values_list) > 100:
        return dict(sorted(values_list.items())[:100])
    return values_list

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERIC FUNCTIONS FOR DATA RETRIEVAL
    # The purpose of this function was to reduce manual coding labor. 
    # I did not want to create a seperate function for each tool
    # The key to this function's versatility is literal eval: I evaluate the real function inside this function
    # Why not just pass the CoinGecko function directly to each tool?:
        # Langchain can only pass in and return a single string: Most Coin Gecko functions have more than one input which may or may not be a string


# Generic function for API-based data retrieval
# func_call_query = a list encapsulated in a string to be literal evaled
def all_purpose_function(func_call_query):
    result = ""
    try:
        func_call_query = eval(func_call_query)
        # func_call_query is a list of 3 things: result = the output of the real coin gecko function call, query = What the user typed in the search engine, num = The numbers of keys to reduce dictionary sizes to
        result = func_call_query[0]
        query = func_call_query[1]
        num = func_call_query[2]
        
        # Num = None if token reduction is not necessary for particular tool
        if num != None:
            # key reduction for a list of dictionaries
            if (type(result) == list):
                if (type(result[0]) == dict):
                    final = []
                    for i in result:
                        new_dict = reduce_dict_to_x(i, query, num)
                        final.append(new_dict)
                    result = final
            # key reduction for a single dictionary
            elif type(result) == dict:
                if len(result) > 1:
                    new_dict = reduce_dict_to_x(result, query, num)
                    result = new_dict
        # res_length = number of tokens after endpoint result filtering 
        res_length = len(str(result))/4
        # Situations for resorting to document search (secon best option) instead of live API request
        if (result == []) or (result == None):
            result = "Try vector search"
        elif res_length > 3800:
            result = "Try vector search"
    except Exception as e:
        print("An error occurred:", e)
        try:
            # Return user query
            result = eval(func_call_query)[1]
        except:
            # Resort to document search
            result =  "Try vector search"
    return result


# Generic function for document based data retrieval
# query = a list encapsulated in a string to be literal evaled: original user query + index (0, 1, 2) - first, second, or third best document retrieval
def vector_search(query):
    try:
        query_output = eval(query)
        query_new = query_output[0]
        index = query_output[1]
        # Before this function is called, there will be a variable called 'option' which determines whether a user is looking up random ('fun stuff') or crypto related content
        if option == 'fun stuff':
            print('worked')
            # Use Google search for 'fun stuff'
            return search.run(query_new)
        else:
        # connect to your url and begin vector search
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
            nearText = {"concepts": [query_new]}
            result = (
                client.query
                .get("Question", ["text", "url"])
                .with_near_text(nearText)
                .with_limit(3)
                .do()
            )
            # Return relevant document based on selected index
            return json.dumps(result["data"]["Get"]["Question"][index], indent=4)
    except:
        # Resort to Google search in times of error
        try:
            return search.run(eval(query)[0])
        except:
            return search.run(query)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOOLS FOR LANGCHAIN
    # Each tool represents a CoinGecko endpoint and instructions for when and how to use it to retrieve appropriate data to answer a user request
    # Each tool has a name: short summary of tool, func: function to retrieve data, and decsription: instructions and examples of how to pass the right input into the function

# CSV conatining list of names and corresponding descriptions for tools
tool_df = pd.read_csv('tools.csv')

# List of tools, can view descriptions in tools.csv
ALL_TOOLS = [
    #/simple/price
    Tool(
        name="Get CURRENT PRICE of a coin given name and currency.",
        func=all_purpose_function,
        description = tool_df.iloc[0, 1]
    ), 
    #/simple/token_price/{id}
    Tool(
    name="Get TOKEN PRICE given token name and platform.",
    func=all_purpose_function,
    description= tool_df.iloc[1, 1]
    ),
    #/coins/{id}/history
    Tool(
        name = "Get HISTORICAL DATA about coin on SPECIFIC DATES.",
        func= all_purpose_function,
        description = tool_df.iloc[2, 1]
    ),
    #/coins/markets
    Tool(
        name = "Get market cap, volume, market cap rank and other market related data given SPECIFIC COIN NAMES and a currency.",
        func = all_purpose_function,
        description = tool_df.iloc[3, 1]
    ),
    #/coins/markets
    Tool(
        name = "Get market cap, volume and other market related data given SPECIFIC COIN RANKS and a currency.",
        func = all_purpose_function,
        description = tool_df.iloc[4, 1]
    ),
    # Find contract address
    Tool(
        name = "Get contract addresses given platform and token.",
        func = search.run,
        description = tool_df.iloc[5, 1]
    ),
    #/coins/list
    Tool(
        name = "Get all supported coins ids, names and symbols.",
        func = all_purpose_function,
        description = tool_df.iloc[6, 1]
    ),
    # /coins/{id} 
    Tool(
        name = "Get any CURRENT DATA about a coin. Make sure to use this before trying any of the vector search tools",
        func = all_purpose_function,
        description = tool_df.iloc[7, 1]
    ),
    #/coins/{id}/market_chart
    Tool(
        name = "Get market data for PAST X NUMBER OF DAYS given coin name, currency, days, and interval.",
        func = all_purpose_function,
        description =  tool_df.iloc[8, 1]
    ),
    #/asset_platforms
    Tool(
        name = "Get list of asset platforms and their chain identifiers. You can pass in optional filter: 'nft'.",
        func = all_purpose_function,
        description = tool_df.iloc[9, 1]                
    ),
    #/coins/categories/list
    Tool(
        name = "Get coin category(s).",
        func= all_purpose_function,
        description = tool_df.iloc[10, 1]
    ),
    #/coins/categories
    Tool(
        name = "Get market data about SPECIFIC COIN CATEGORIES.",
        func = all_purpose_function,
        description = tool_df.iloc[11, 1]    
    ),
    # Alternative Search options for when API endpoints fail: document search
        # ALL top 3 documents together would surpass the token limit, so we have broken it into 3 tools for each
        # LangChain will prioritize 'Document search number 1" and will use others if it is not able to get to a relevant answer
    Tool(
        name="Document search number 1",
        func=vector_search,
        description = tool_df.iloc[12, 1]
    ),
    Tool(
        name="Document search number 2",
        func=vector_search,
        description = tool_df.iloc[13, 1]
    ),
    Tool(
        name="Document search number 3",
        func=vector_search,
        description = tool_df.iloc[14, 1]
    )
]
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOOL FILTERING & RETRIEVAL
    # When langchain reads through descriptions for all of the tools, it passes the token limit and causes the langchain session to break
    # To prevent this, we are using a vector store to filter tool selection


# Puts tool descriptions into a document format
docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS[:-3])]
# Converts documents into embeddings
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())


# function to retrieve tools by doing similarity search of tool description with user request and returns relevant tools accordingly
def get_tools(query):
    docs = vector_store.similarity_search(query)
    # We want vector search tools by default, which is why I did not include them in the vector_store for tool retrieval
    return [ALL_TOOLS[d.metadata["index"]] for d in docs] + [ALL_TOOLS[-3]] + [ALL_TOOLS[-2]] + [ALL_TOOLS[-1]]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PROMPT TEMPLATE FOR LANGCHAIN THOUGHTS
    # This dictates how the respective langchain agent will think and what actions it will take when and how it will format answers

# Prompt template
template = """Answer the following questions as best you can:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action. It must be in this EXACT format: [function call, original prompt, an int]. Do not add any extra quotations in function call besides what is described in tool description!!!
Observation: the result of the action. 
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Format answer so that it is easy to read. Bold key words!!! Use colons and NEWLINE CHARACTERS!!! For example, if user asks about total market volume and ath of Ethereum. Format Final Answer as such:
    1. total market volume: ...\n2. ath: ...\n
Make agent executor output also retain the newline characters
Remember to the stick to the exact format described in the tool descriptions!
Question: {input}
{agent_scratchpad}"""


# Default class from LangChain for using custom prompt templates for custom tool retrieval
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

# The final prompt template that gets passed into the llmchain takes in:
    #1) template (NL instructions for flow of thought and how to format answers)
    #2) tools_getter: function for retrieving tools
    #3) input_variables: options important for formatting and using custom prompt template
prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CUSTOM OUTPUT PARSER: HOW LANGCHAIN AGENT WILL ARRIVE AT A FINAL ANSWER

# Default class from LangChain for using custom output parsing for custom tool retrieval
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CREATING LANGCHAIN AGENT
    # An executor that will connect to an llm dictate what data it will use to answer user questions

# Large Language Model Selection
llm = OpenAI(temperature=0)

# Chain that takes in an llm & instructions for what kind of thought process it will use 
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in ALL_TOOLS]

# Defining agent which takes in llm, prompt template, output parser, list of tool names, and instructions for when to stop running in prompt template
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

# Defining agent executor which encaspsulates inctructions for running the agent
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=ALL_TOOLS, verbose=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# STREAMLIT APP
    # UI for search engine

st.title("Coin Gecko API Integration With Vector Search\n")
st.caption('_:blue[API integration support: coin price, token price, coin history, current coin market data, coin market charts, asset platforms, crypto categories and their market data, and vector search\n]_', 
unsafe_allow_html=True)
col1, col2 = st.columns([1, 3])
with col1:
    option = st.selectbox('What kind of information?',('subject matter', 'fun stuff'))
with col2:
    user_prompt = st.text_input("Enter a question. Do not use contractions!", "")
submit = st.button("Submit")
if submit:
    st.write('''''' + agent_executor.run(user_prompt))
