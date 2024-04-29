from dotenv import load_dotenv
import os
import pandas as pd  
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str,context
from note_engine import note_engine
from llama_index.core.tools  import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from pdf import major_engine

load_dotenv()

data_path = os.path.join("data", "school.csv")
data_df= pd.read_csv(data_path)

data_query_engine = PandasQueryEngine(df=data_df, verbose=True, instruction_str=instruction_str)  # false if i dont want to see the thought process
#optimize  performance
data_query_engine.update_prompts({"pandas_prompt" : new_prompt})
tools =[
    note_engine,
    QueryEngineTool(query_engine=data_query_engine,
    metadata=ToolMetadata(
        name="data_data",
        description="this gives information about students in ESSAT",
        ),
    ),
    QueryEngineTool(query_engine=major_engine,
    metadata=ToolMetadata(
        name="major_data",
        description="this gives detailed information about your major in ESSAT",
        ),
    ),
]
llm = OpenAI(model="gpt-3.5-turbo-0613")
agent=ReActAgent.from_tools(tools, llm=llm, verbose=True,context=context)# false if i dont want to see the thought process

while(prompt := input("Enter a prompt(question) (q to quit)  ")) != "q":
    result = agent.query(prompt)
    print (result)

