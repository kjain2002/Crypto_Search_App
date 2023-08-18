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
import streamlit as st
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
# from langchain.utilities import RequestsWrapper
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

os.environ['OPENAI_API_KEY'] = "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"
os.environ["SERPER_API_KEY"] = "d4247b1d0eadad3649101d95e6185c569b02bf19"
openai.api_key = "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"

search = GoogleSerperAPIWrapper()
cg = CoinGeckoAPI()

# Side functions
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
def get_market_data_by_rank(currency, ranks):
    result = []
    dic = cg.get_coins_markets(currency)
    for coin in dic:
            if coin['market_cap_rank'] in ranks:
                result.append(coin)

    return result

def get_coins_list(label = "", starts_with = ""):
    values_list = {}
    for dict_item in cg.get_coins_list():
        if (label in dict_item) and (dict_item[label].lower().startswith(starts_with.lower())):
            values_list[dict_item['id']]= dict_item[label]
    if len(values_list) > 100:
        return dict(sorted(values_list.items())[:100])
    return values_list

# Main function
def all_purpose_function(func_call_query):
    os.environ['OPENAI_API_KEY'] = "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"
    os.environ["SERPER_API_KEY"] = "d4247b1d0eadad3649101d95e6185c569b02bf19"
    result = ""
    try:
        func_call_query = eval(func_call_query)
        result = func_call_query[0]
        query = func_call_query[1]
        num = func_call_query[2]
        if num != None:
            if (type(result) == list):
                if (type(result[0]) == dict):
                    final = []
                    for i in result:
                        new_dict = reduce_dict_to_x(i, query, num)
                        final.append(new_dict)
                    result = final
            elif type(result) == dict:
                if len(result) > 1:
                    new_dict = reduce_dict_to_x(result, query, num)
                    result = new_dict
        res_length = len(str(result))/4
        if (result == []) or (result == None):
            result = "Try vector search"
        elif res_length > 3800:
            result = "Try vector search"
    except Exception as e:
        print("An error occurred:", e)
        try:
            result = eval(func_call_query)[1]
        except:
            result =  search.run(func_call_query)
    return result
ALL_TOOLS = [
    #/simple/price
    Tool(
        name="Get CURRENT PRICE of a coin given name and currency.",
        func=all_purpose_function,
        description = '''Put in call in this format [cg.get_price(coin_name_not_symbol, currency), original query, None]
                         
                         Sample user input -> "What is the price of Ethereum in usd?"
                         Sample Action input -> [cg.get_price('ethereum', 'usd'), 'What is the price of Ethereum in usd?', None]

                         Sample user input -> "What is the price of Ethereum and Dogecoin in usd?"
                         Sample Action input -> [cg.get_price(['ethereum', 'dogecoin'], 'usd'), 'What is the price of Ethereum and Dogecoin in usd?', None]

                         Sample user input -> "What is the price of Ethereum and Dogecoin in Chinese and United States currency?"
                         Sample Action input -> [cg.get_price(['ethereum', 'dogecoin'], ['usd', 'cny']), 'What is the price of Ethereum and Dogecoin in Chinese and United States currency?', None]

                         if information derived from observation = None or 0.0. Say "Data is not available in CoinGecko"'''
    ), 
    #/simple/token_price/{id}
    Tool(
    name="Get TOKEN PRICE given token name and platform.",
    func=all_purpose_function,
    description='''Put in call in this format [cg.get_token_price(id, contract_address, currency), original query, None]
                     Sample user input -> "What is the price of the ADA token on Cardano in USD?"
                     Find contract address of respective token name on respective platform using Google Search
                     Sample Action input -> [cg.get_token_price('cardano', '0x3ee2200efb3400fabb9aacf31297cbdd1d435d47', 'usd'), 'What is the price of the ADA token on Cardano in USD?', None]
                     
                     Sample user input -> "What is the price of DAI token on Ethereum blockchain platform in Europe?"
                     Find contract address of respective token name on respective platform using Google Search
                     Sample Action input -> [cg.get_token_price('ethereum', '0x6B175474E89094C44Da98b954EedeAC495271d0F', 'eur'), 'What is the price of DAI token on Ethereum blockchain platform in Europe?', None]
                     
                     Sample user input -> "What is the price of the token with this corresponding address: 0x6B175474E89094C44Da98b954EedeAC495271d0F on Ethereum blockchain in chinese currency?"
                     Sample Action input -> [cg.get_token_price('ethereum', '0x6B175474E89094C44Da98b954EedeAC495271d0F', 'cny'), 'What is the price of the token with this corresponding address: 0x6B175474E89094C44Da98b954EedeAC495271d0F on Ethereum blockchain in chinese currency?', None]
                     
                     Sample user input -> "What is the price of the DAI, Tether, and Binance Coin on Ethereum blockchain in USD, CNY, and Eur currencies?"
                     Find respective contract addresses of respective token names on respective platform using Google Search 
                     Sample Action input -> [cg.get_token_price('ethereum', ['0x6B175474E89094C44Da98b954EedeAC495271d0F','0xdac17f958d2ee523a2206206994597c13d831ec7','0xB8c77482e45F1F44dE1745F52C74426C631bDD52'] , ['usd', 'cny', 'eur']), 'What is the price of the DAI, Tether, and Binance Coin on Ethereum blockchain in USD, CNY, and Eur currencies?', None]
                     
                     if information derived from observation = None or 0.0. Say "Data is not available in CoinGecko"'''
    ),
    #/coins/{id}/history
    Tool(
        name = "Get HISTORICAL DATA about coin on SPECIFIC DATES.",
        func= all_purpose_function,
        description = '''Put in call in this format: [cg.get_coin_history_by_id(coin_name, "dd-mm-yyyy"), original query, 3]
                         Sample user input -> "What was the price of Ethereum on April 4th 2023?"
                         Sample Action input -> [cg.get_coin_history_by_id('ethereum', '05-04-2023'), "What was the price of Ethereum on April 4th 2023?", 3]
                         Sample user input -> "Get Dogecoin symbol yesterday"
                         Sample Action input -> [cg.get_coin_history_by_id('dogecoin', date = (dt.datetime.today() - timedelta(days=1)).strftime('%d-%m-%Y')), 'Get Dogecoin symbol yesterday', 3] 
                         Sample user input -> "Summarize market data about Ethereum 3 days ago in usd"
                         Sample Action input -> [cg.get_coin_history_by_id('ethereum', date = (dt.datetime.today() - timedelta(days=3)).strftime('%d-%m-%Y'))['market_data], 'Summarize market data about Ethereum have 3 days ago in usd', 3]
                         if information derived from observation = None or 0.0. Say "Data is not available in CoinGecko"'''
    ),
    #/coins/markets
    Tool(
        name = "Get market cap, volume, market cap rank and other market related data given SPECIFIC COIN NAMES and a currency.",
        func = all_purpose_function,
        description = '''Put in call in this format [cg.get_coins_markets(vs_currency = currencies, ids = coin_names), original_query, None]
                         Sample user input -> "Get Ethereum market data"
                         Sample Action input -> [cg.get_coins_markets(vs_currency = 'usd', ids = 'ethereum'), 'Get Ethereum market data', None]
                         Sample user input -> "Get Ethereum market data in usd"
                         Sample Action input -> [cg.get_coins_markets(vs_currency = 'usd', ids = 'ethereum'), 'Get Ethereum market data in usd', None]
                         Sample user input -> "Dogecoin and Ethereum market data in chinese currency"
                         Sample Action input -> [cg.get_coins_markets(vs_currency = 'cny', ids = ['dogecoin', 'ethereum']), 'Dogecoin and Ethereum market data in chinese currency', 5] 
                         if information derived from observation = None or 0.0. Say "Data is not available in CoinGecko"'''
    ),
    #/coins/markets
    Tool(
        name = "Get market cap, volume and other market related data given SPECIFIC COIN RANKS and a currency.",
        func = all_purpose_function,
        description = '''Put in this call format : [get_market_data_by_rank(currency_name, [...list_of_ranks...]), original query, 5]
                        Sample user input -> "List coins of top 5 market cap ranks"
                        Sample Action input-> [get_market_data_by_rank('usd', [1, 2, 3, 4, 5]), 'List coins of top 5 market cap ranks', 5]
                         
                        Sample user input -> "List coins of top 5 market cap ranks in usd"
                        Sample Action input-> [get_market_data_by_rank('usd', [1, 2, 3, 4, 5]), 'List coins of top 5 market cap ranks in usd', 5]  
                        
                        Sample user input -> "List names and summarize market data in chinese currency of top 6 crypto coins by market cap ranks"
                        Sample Action input-> [get_market_data_by_rank('cny', [1, 2, 3, 4, 5, 6]), 'List names and summarize market data in chinese currency of top 6 crypto coins by market cap ranks', 5]
                        
                        if information derived from observation = None or 0.0. Say "Data is not available in CoinGecko"'''
    ),
    # Find contract address
    Tool(
        name = "Get contract addresses given platform and token.",
        func = search.run,
description = "Find contract address that corresponds to a blockchain network and a token"),
    #/coins/list
    Tool(
        name = "Get all supported coins ids, names and symbols.",
        func = all_purpose_function,
        description = '''Put in call in this format: [get_coins_list(label, starts_with), original query, None]
                      Sample user input -> "List all supported coins starting with b"
                      Sample action input -> [get_coins_list(label = 'name', starts_with = 'b'), 'List all supported coins starting with b', None]

                      Sample user input -> "List all supported coin symbols starting with b"
                      Sample action input -> [get_coins_list(label = 'symbol', starts_with = 'b'), 'List all supported coin symbols starting with b', None]

                      Sample user input -> "Is ethereum a supported coin name?"
                      Sample action input -> [get_coins_list(label = 'name', starts_with = 'ethereum'), 'Is ethereum a supported coin name?', None]

                      Sample user input -> "Is zoc a supported coin symbol?"
                      Sample action input -> [get_coins_list(label = 'symbol', starts_with = 'zoc'), 'Is zoc a supported coin symbol?', None]
                      
                      if information derived from observation = None or 0.0. Say "Data is not available in CoinGecko"'''
    ),
    # /coins/{id} 
    Tool(
        name = "Get CURRENT DATA about a coin.",
        func = all_purpose_function,
        description = '''Put in call in this format [cg.get_coin_by_id(coin_name), original_query, 3]
        Sample user input -> "Get developer data about bitcoin"
        Sample action input -> [[cg.get_coin_by_id(coin_name), 'Get developer data about bitcoin', 3]]'''),
    #/coins/{id}/market_chart
    Tool(
        name = "Get market data for PAST X NUMBER OF DAYS given coin name, currency, days, and interval.",
        func = all_purpose_function,
        description =  '''Put in call in this format [cg.get_coin_market_chart_by_id(coin_name, currency, days, interval), original_query, None] 
                          
                          Sample user input -> "Get Bitcoin market data for the past 2 days"
                          Sample action input -> [cg.get_coin_market_chart_by_id('bitcoin', 'usd', 2, interval = 'daily'), 'Get Bitcoin market data for the past 2 days', None]
                          
                          Sample user input -> "Get Ethereum market data for the past 3 days in chinese currency"
                          Sample Action input -> [cg.get_coin_market_chart_by_id('ethereum', 'cny', 3, interval = 'daily'), original_query, None]'''
    ),
    #/asset_platforms
    Tool(
        name = "Get list of asset platforms and their chain identifiers. You can pass in optional filter: 'nft'.",
        func = all_purpose_function,
        description = '''Pass in call in one of these formats:
                            1) [top_dics(cg.get_asset_platforms(), 'name', original query, 10), original query, None]
                            2) [top_dics(cg.get_asset_platforms(), 'chain_identifier', original query, 10), original query, None]
                            3) [cg.get_asset_platforms(filter = 'nft'), original query, None]
                         
                         Sample user input -> "What is the chain identifier for the binance chain?"
                         Sample Action input -> [top_dics(cg.get_asset_platforms(), 'name', 'What is the chain identifier for the binance chain?', 10), 'What is the chain identifier for the binance chain?', None]
                          
                         Sample user input -> "What is the name of the platform with chain identifier 50?"
                         Sample Action input -> [top_dics(cg.get_asset_platforms(), 'chain_identifier', 'What is the name of the platform with chain identifier 50?', 10), 'What is the name of the platform with chain identifier 50?', None]
                         
                         Sample user input -> "Which asset platforms support nfts?"
                         Sample Action input -> [cg.get_asset_platforms(filter = 'nft'), 'Which asset platforms support nfts?', None]'''                
    ),
    #/coins/categories/list
    Tool(
        name = "Get coin category(s).",
        func= all_purpose_function,
        description = '''Pass in call in this exact format: [top_dics(cg.get_coins_categories_list(), 'name', original query, 10), original query, None]
                         
                         Sample user input -> "Which crypto categories pertain to tech?"
                         Sample Action input -> [top_dics(cg.get_coins_categories_list(), 'name', 'Which crypto categories pertain to tech?', 10), 'Which crypto categories pertain to tech?', None]

                         Sample user input -> "Coin categories related to ethereum"
                         Sample Action input -> [top_dics(cg.get_coins_categories(), 'name', 'Coin categories related to ethereum', 10), 'Coin categories related to ethereum', None]'''
    
    ),
    #/coins/categories
    Tool(
        name = "Get market data about SPECIFIC COIN CATEGORIES.",
        func = all_purpose_function,
        description = '''Pass in call in one of these EXACT formats: 
                            1) [cg.get_coins_categories()[:15], originial query, 2]
                            2) [cg.get_coins_categories(order = 'market_cap_asc')[:15], original query, 2]
                            3) [cg.get_coins_categories(order = 'market_cap_change_24h_desc')[:15], original query, 2]
                            4) [cg.get_coins_categories(order = 'market_cap_change_24h_asc')[:15], original query, 2]
                         Sample user input -> "List coin categories of top 10 market caps?"
                         Sample Action input -> [cg.get_coins_categories()[:15], 'List coin categories of top 10 market caps?', 2]
                        
                         Sample user input -> "List coin categories with 10 lowest market caps?"
                         Sample Action input -> [cg.get_coins_categories(order = 'market_cap_asc')[:15], 'List coin categories with 10 lowest market caps?', 2]
                        
                         Sample user input -> "List coin categories of top 10 market cap incresaes in the last 24 hours?"
                         Sample Action input -> [cg.get_coins_categories(order = 'market_cap_change_24h_desc')[:15], 'List coin categories of top 10 market cap incresaes in the last 24 hours?', 2]
                        
                         Sample user input -> "List coin categories of top 10 market cap decreases in the last 24 hours?"
                         Sample Action input -> [cg.get_coins_categories(order = 'market_cap_change_24h_asc')[:15], 'List coin categories of top 10 market cap decreases in the last 24 hours?', 2]'''

    )
    # #Google Search
    # Tool(
    #     name = "Google search",
    #     func = search.run,
    #     description = '''Resort to this if tools don't work out
    #                      Pass in call in this EXACT format: original query

    #                      Sample user input -> What is the difference between a token and a coin?
    #                      Sample Action input -> What is the difference between a token and a coin?

    #     '''
    # )
]


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
                            return "Try Google Search"
                        
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
    llm = OpenAI(temperature=0)
    st.write("Open AI Agen acquired",llm)
    create_call_chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=True
    )
    
    return create_call_chain.run(api_call=api_call, query=query)
class AllPurposeTool(BaseTool):
    name = "AllPurposeTool"
    description = """
    Useful for when you need to access real time crypto data from the DefiLlama API.
    Pass in the user's question as the query in all lowercase."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        search_term_vector = get_embedding(query, engine='text-embedding-ada-002')
        df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
        top_calls = np.array(df.sort_values('similarities', ascending=False).head(3)['call prefixes'])

        for i in range(len(top_calls)):
            if '{' in top_calls[i]:
                complete_call = construct_call(top_calls[i], query)
                call_test = requests.get(complete_call)
            else:
                complete_call = top_calls[i]
                call_test = requests.get(complete_call)

            if call_test.status_code == 200:
                final_call = complete_call
                return all_purpose_defi(repr([final_call, query, 2]))

        return "Try the google tool"
        
    
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

df = pd.read_csv('df.csv')

def gecko_code(query_final):
    # def get_tools(query_final):
    #     return selected_tools
    docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # retriever = vector_store.as_retriever()

    def get_tools(query):
        docs = vector_store.similarity_search(query)
        return [ALL_TOOLS[d.metadata["index"]] for d in docs] + [ALL_TOOLS[-1]]

    template = """Answer the following questions as best you can:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action. If tool name is 'Google search' only pass in the original query. Otherwise, it must be in this EXACT format: [function, original prompt, an int].
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Remember to the stick to the exact format described in the tool descriptions! And resort to google search if the tools don't work out.
    Question: {input}
    {agent_scratchpad}"""

    from typing import Callable
    # Set up a prompt template
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
    prompt = CustomPromptTemplate(
        template=template,
        tools_getter=get_tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
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
    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in ALL_TOOLS]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=ALL_TOOLS, verbose=True)

    # st.title("Coin Gecko API Integration")
    # user_prompt = st.text_input("Enter a question. Do not use contractions!", "")
    # submit = st.button("Submit")
    # if submit:
    st.write(agent_executor.run(query_final))
    # agent_executor.run(query_final)
    # if submit:
    #     st.write(agent_executor.run(query_final))


def defi_code(query_final):
    search = GoogleSerperAPIWrapper()
    tools = [
        AllPurposeTool(),
        BridgeTool(),
        Tool(
            name="Google Search",
            func=search.run,
            description="Only use if all the other tools error"
        )
    ]

    prefix = """You are an Artificial Intelligence with the purpose of answering questions regarding web3 topics. You have access to the following tools:"""
    suffix = """Begin!"

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "agent_scratchpad"]
    )
    llm = OpenAI(temperature=0)
    st.write("Open AI Agen acquired",llm)
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

    agent_executor.run(query_final)
    st.write(agent_executor.run(query_final))


def get_call(query):
    ian_docs = [Document(page_content = data[0] + " " + data[1], metadata = {"index": i, "label": "defi"}) for i, data in enumerate(zip(df['call descriptions'], df['call prefixes']))]
    docs = [Document(page_content=t.name, metadata={"index": i, "label": "gecko"}) for i, t in enumerate(ALL_TOOLS)]
    ian_docs.extend(docs)
    vector_store = FAISS.from_documents(ian_docs, OpenAIEmbeddings())
    retriever = vector_store.similarity_search(query)
    most_relevant = retriever[0].metadata["label"]
    st.write(retriever)
    return most_relevant
# query_final =  "weekly volume of stargate?"
st.title("Coin Gecko API Integration")
user_prompt = st.text_input("Enter a question. Do not use contractions!", "")
submit = st.button("Submit")
if submit:
    # call = get_call(user_prompt)
    call = 'gecko'
    if call == 'gecko':
        st.write("API Integration Selection: Coin Gecko")
        # gecko_code(user_prompt)
        docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        # retriever = vector_store.as_retriever()

        def get_tools(query):
            docs = vector_store.similarity_search(query)
            return [ALL_TOOLS[d.metadata["index"]] for d in docs] + [ALL_TOOLS[-1]]

        template = """Answer the following questions as best you can:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action. If tool name is 'Google search' only pass in the original query. Otherwise, it must be in this EXACT format: [function, original prompt, an int].
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Remember to the stick to the exact format described in the tool descriptions! And resort to google search if the tools don't work out.
        Question: {input}
        {agent_scratchpad}"""

        from typing import Callable
        # Set up a prompt template
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
        prompt = CustomPromptTemplate(
            template=template,
            tools_getter=get_tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
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
                # Return the action and action input``
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        output_parser = CustomOutputParser()
        llm = OpenAI(temperature=0)
        st.write("Open AI Agen acquired", llm)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in ALL_TOOLS]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=ALL_TOOLS, verbose=True)
        output = agent_executor.run(user_prompt)
        st.write(output)
    else:
        st.write("API Integration Selection: DefiLlama")
        defi_code(user_prompt)


