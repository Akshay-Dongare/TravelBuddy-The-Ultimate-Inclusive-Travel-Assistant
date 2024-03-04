# Relevance AI html code
#<iframe src="https://app.relevanceai.com/form/bcbe5a/efce2479-8da1-4a3c-befa-e40679bf1b41" width="500px" height="750px" frameborder="0"></iframe>


import Utils.config as config
from flask import Flask, render_template, request
import os
#from openai import OpenAI
from langchain import hub
import speech_recognition as sr
import pyttsx3
from langchain.agents import Tool, AgentExecutor, create_react_agent, AgentOutputParser,Tool
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

app = Flask(__name__)

template = """Answer the following questions as best you can, but speaking as passionate travel expert. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a detailed day by day final answer to the original input question

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer.


Question: {input}
{agent_scratchpad}"""

template_with_history = """Answer the following questions as best you can, but speaking as passionate travel expert. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer.

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}"""
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

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
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


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
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_online(input_text):
    search = DuckDuckGoSearchRun().run(f"site:tripadvisor.com {input_text}")
    return search

def search_hotel(input_text):
    search = DuckDuckGoSearchRun().run(f"site:booking.com {input_text}")
    return search

def search_flight(input_text):
    search = DuckDuckGoSearchRun().run(f"site:skyscanner.com {input_text}")
    return search

def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search

tools = [

    Tool(
        name="Search general",
        func=search_general,
        description="useful for when you need to answer general travel questions"
    ),
    Tool(
        name="Search tripadvisor",
        func=search_online,
        description="useful for when you need to answer trip plan questions"
    ),
    Tool(
        name="Search booking",
        func=search_hotel,
        description="useful for when you need to answer hotel questions"
    ),
    Tool(
        name="Search flight",
        func=search_flight,
        description="useful for when you need to answer flight questions"
    )

]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps",'tool_names', 'tools', 'agent_scratchpad']
)
prompt_with_history = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history",'tool_names', 'tools', 'agent_scratchpad']
)
prompt_langchain_hub = hub.pull("hwchase17/react-chat")
output_parser = CustomOutputParser()
memory = ConversationBufferWindowMemory(k=5)
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-0613", openai_api_key=config.OPENAI_API_KEY)
# LLM chain consisting of the LLM and a prompt
#llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)#choose if you want prompt with history or just prompt
tool_names = [tool.name for tool in tools]
'''agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)'''
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_with_history)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory,handle_parsing_errors=True)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/voice',methods=["GET", "POST"])
def voice():
    # Initialize the recognizer 
    r = sr.Recognizer() 

    # Function to convert text to
    # speech
    def SpeakText(command):
        
        # Initialize the engine
        engine = pyttsx3.init("sapi5", True) #driver for windows and debug mode as args
        engine.say(command) 
        engine.runAndWait()

    # Loop infinitely for user to
    # speak
    
    while(1):    
        
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input 
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
    
                print("Did you say ",MyText)
                
                #Q&A using gpt3.5
                completion = agent_executor.invoke({"input":MyText})
                '''
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", 
                        "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{MyText} to the best of my knowledge."
                        }
                        ],
                model="gpt-3.5-turbo",
                )
                SpeakText(completion.choices[0].message.content)'''
                SpeakText(completion['output'])
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")

def response_by_llm(query):
    #Q&A using gpt3.5
    '''
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", 
             "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{query} to the best of my knowledge."
             }
             ],
    model="gpt-3.5-turbo",
    )'''
    completion = agent_executor.invoke({"input":query})
    return completion['output']

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return response_by_llm(input)

if __name__ == '__main__':
    app.run(debug=True)