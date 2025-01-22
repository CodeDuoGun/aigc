import getpass
import os

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

def set_env():
    """"""
    # os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    os.environ["LANGCHAIN_TRACING_V2"] ="true"
    os.environ["LANGCHAIN_API_KEY"] ="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"
    os.environ["LANGCHAIN_TRACING_V2"] ="true"
set_env()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str # 提示模版中有两个参数，因此自定义类新增参数

class LocalLangchain():
    def __init__(self) -> None:
        """
            export OPENAI_API_KEY=<your-openai-api-key>
        """
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.model.invoke("Hello, world!")
        
    def add_memory(self):
        # Define a new graph
        workflow = StateGraph(state_schema=State)

        # Define the function that calls theself.model
        # 这个func 还可以异步
        def call_model(state: State):
            prompt = self.build_chatbot_prompt().invoke(state)
            response = self.model.invoke(prompt)
            # response =self.model.invoke(state["messages"])
            return {"messages": response}

        # Define the (single) node in the graph
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        # Add memory
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        self.invoke_resp(app)

    def invoke_resp(self,app):
        config = {"configurable": {"thread_id": "abc123"}}
        query = "What's my name?"
        language = "English"

        input_messages = [HumanMessage(query)]
        output = app.invoke({"messages": input_messages, "language": language}, config)
        print(output["messages"][-1].pretty_print())
        return output["messages"][-1].pretty_print() 

    def invoke_stream_invoke(self, app):
        """
        
        """
        config = {"configurable": {"thread_id": "abc789"}}
        query = "Hi I'm Todd, please tell me a joke."
        language = "English"

        input_messages = [HumanMessage(query)]
        for chunk, metadata in app.stream(
            {"messages": input_messages, "language": language},
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):  # Filter to just model responses
                print(chunk.content, end="|")
    

    def build_a_llm(self):
        #self.model 选择
        # prompt 给出角色
        messages = [
            SystemMessage("Translate the following from English into Italian"),
            HumanMessage("hi!"),
        ]
        # 结果解析
        self.model.invoke(messages) #orself.model.invoke([{"role": "user", "content": "Hello"}]) orself.model.invoke([HumanMessage("Hello")])
        prompt = self.build_a_prompt()
        response =self.model.invoke(prompt)
        print(response.content)
        # langsmith 追踪你的数据

        # langserve 部署

    def build_a_chatbot(self):
        """
        """
        # 创建model
        

        # chat memory 
        self.add_memory()

        # invoke


        pass

    def build_a_agent(self):
        """"""
        pass
        
    def build_translate_prompt(self):
        """定义一个提示模版"""
        system_template = "Translate the following from English into {language}"

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{text}")]
        )
        prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
        return prompt

    def build_chatbot_prompt(self):
        """"""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You talk like a pirate. Answer all questions to the best of your ability in {language}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt_template