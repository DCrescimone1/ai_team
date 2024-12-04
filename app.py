import os
from typing import Annotated
import yaml
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from autogen import UserProxyAgent, AssistantAgent, ConversableAgent, register_function
import autogen
# from pyairtable import Api
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from pathlib import Path


load_dotenv()
scraping_api_key = os.getenv("SCRAPING_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")
config_list = [
    {
    "model": os.getenv("OPENAI_MODEL"),
    "api_key": os.getenv("OPENAI_API_KEY")
    }
]
airtable_api_key = os.getenv("AIRTABLE_API_KEY")

# model gpt-4o gpt-4-turbo-preview gpt-3.5-turbo-16k-0613

# import Agents Prompt from yaml file

with open('prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

researcher_description = prompts['researcher']['description']
researcher_prompt = prompts['researcher']['prompt']
project_manager_description = prompts['project_manager']['description']
project_manager_prompt = prompts['project_manager']['prompt']
publisher_description = prompts['publisher']['description']
publisher_prompt = prompts['publisher']['prompt']
publisher_manager_description = prompts['publisher_manager']['description']
publisher_manager_prompt = prompts['publisher_manager']['prompt']
fed_description = prompts['fed']['description']
fed_prompt = prompts['fed']['prompt']
fed_manager_description = prompts['fed_manager']['description']
fed_manager_prompt = prompts['fed_manager']['prompt']


# ---------- CREATE A FUNCTION ------------- #

# Function for google search
def google_search(
        search_keyword: Annotated[str, "Optimal search keywords are those most likely to yield relevant results for the information you seek."]
        ) -> Annotated[str, "Response from  google search"]:
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': serp_api_key, 
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text

# Function for scraping
def summary(objective, content):
    llm = ChatOpenAI(temperature =0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose= True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)
    
    return output

def web_scraping(
        objective: Annotated[str, "the goal of scraping the website. e.g. any specific type of informtion you are looking for?"], 
        url: Annotated[str, "the url website you want to scrape"]
        ) -> Annotated[requests.Response, "to send the post request"]:
    # scrape website, and also will summarize the content based on objective
    # objective is the original objective & task that user give to the agent

    print(f"Scraping website...{url}\n\nAnd this is the objective {objective}\n\n")


    # define the headers for the request
    headers= {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }

    payload = {
        "api_key": scraping_api_key,
        "url": url,
        "headers": json.dumps(headers),
    }

    # define the data to be sent in the request
    data = {
        "url": url
    }

    # convert Python object to JSON string
    data_json =json.dumps(data)

    # send the POST request
    response = requests.post(f"https://scraping.narf.ai/api/v1/", params=payload) 

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# Function for get airtable records
def get_airtable_records(base_id: str, table_id: str):
    # url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    url = f"https://api.airtable.com/v0/appHkDdlfDhiwQF8c/tblLYWlmKeHOHgfFn"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
    }   

    response = requests.request("GET", url=url, headers=headers)
    data = response.json()
    return data     

# Function for update airable records
def update_single_airtable_record(id: str, fields: str):
    # url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    url = f"https://api.airtable.com/v0/appHkDdlfDhiwQF8c/tblLYWlmKeHOHgfFn"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
        'Content-Type': "application/json"
    }  

    data = {
        "records": [{
            "id": id,
            "fields": fields
      }]
    }

    response = requests.patch(url=url, headers=headers, data=json.dumps(data))
    data = response.json()
    return data 

# ---------- CREATE AGENT ------------- #

# Create researcher agent
researcher = ConversableAgent(
    name = "researcher",
    system_message=researcher_prompt,
    description=researcher_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create research manager agent
project_manager = AssistantAgent(
    name="project_manager",
    system_message=project_manager_prompt,
    description=project_manager_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create Publisher
publisher = ConversableAgent(
    name="publisher",
    system_message=publisher_prompt,
    description=publisher_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create Publisher_Manager
publisher_manager = AssistantAgent(
    name="publisher_manager",
    system_message=publisher_manager_prompt,
    description=publisher_manager_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create front end developer (FED)
fed = AssistantAgent(
    name="fed",
    system_message=fed_prompt,
    description=fed_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create front end developer manager (FED)
fed_manager = AssistantAgent(
    name="fed_manager",
    system_message=fed_manager_prompt,
    description=fed_manager_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4o", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

executor = LocalCommandLineCodeExecutor(
    timeout=10,
    work_dir=Path("output"),
)


# code_executor = ConversableAgent(
#     name = "code_executor",
#     system_message=researcher_prompt,
#     description=researcher_description, 
#     llm_config=False,
#     code_execution_config={
#         "executor": executor,
#     },
#     human_input_mode="NEVER"
# )

# Create user proxy
user_proxy = UserProxyAgent(name="user_proxy", description="User Proxy Agent",
     is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
     human_input_mode="NEVER",
     max_consecutive_auto_reply=3,
     code_execution_config={"executor": executor}
     )


register_function(
    google_search,
    caller=researcher,
    executor=researcher,
    name="google_search",
    description="Google Search to return results of research keywords"
)

register_function(
    web_scraping,
    caller=researcher,
    executor=researcher,
    name="web_scraping",
    description="scrape website content based on url"
)


# create the Speaker selectipon method:
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is user_proxy:
        # print("\n\n\n\n\n")
        # print(messages[-1])
        # print("\n\n\n\n\n")
        if "<!DOCTYPE html>" in messages[-1]["content"] or "exitcode: " in messages[-1]["content"]:
           return None
       # init -> retrieve (director)
        return researcher
    elif last_speaker is researcher:
       if "tool_calls" in messages[-1]:
           return researcher
       else: return publisher
    # elif last_speaker is project_manager:
    #    if messages[-1]["content"] == "Well done! Research completed.":
    #        # Research --(research compleated)--> Results
    #        return publisher
    #    else:
    #        # Research --(Research incompleated)--> Researcher
    #        return researcher
    elif last_speaker is publisher:
       return publisher_manager
    elif last_speaker is publisher_manager:
        return fed
    elif last_speaker is fed:
        return fed_manager
    elif last_speaker is fed_manager:
        if "Well done! The HTML is good to go." in messages[-1]["content"]:
            # Research --(research compleated)--> Results
            return user_proxy
        else:
            # Research --(Research incompleated)--> Researcher
            return fed
    

# Create group chat
groupchat = autogen.GroupChat(
    # agents= [user_proxy,researcher,project_manager,publisher, publisher_manager, fed, fed_manager], 
    agents= [user_proxy,researcher,publisher, publisher_manager, fed, fed_manager], 
    messages=[], 
    max_round=35, 
    speaker_selection_method=state_transition) # use "auto" to exploit LLM to automatically select the next speeker or "round_robin" to follow a cascade flow

group_chat_manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config={
        "config_list":config_list})

# ----------- START CONVERSATION --------- #
user_proxy.initiate_chat(group_chat_manager, message="base yourself on the last news about drought in sicily and days in which the people were without wate,r find a optimal ammount of days to cover with the tank. Use liters not gallons")
# print(researcher.llm_config["tools"])