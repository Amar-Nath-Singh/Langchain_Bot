from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.chains import LLMMathChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun



llm = ChatOllama(model='llama3.1:8b-instruct-fp16', base_url="http://10.21.139.236:11434", temperature=0)
embedding_function = OllamaEmbeddings(model="nomic-embed-text", base_url="http://10.21.139.236:11434")
persist_directory = "Book_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
search_tool = DuckDuckGoSearchRun()
problem_chain = LLMMathChain.from_llm(llm=llm)

tools = [
    Tool(
        name='Langchain Book',
        func=qa.run,
        description=(
            'Use this tool when answering questions related to Langchain and basic machine learning. This tool provides documentation of Langchain library and sklearn.'
        )
    ),
    Tool(
    name='Internet Search',
    func=search_tool.invoke,
    description=(
    'Use this tool when you need to gather general information from the internet to answer a question or provide context.'
    )
    ),
    Tool(name="Calculator", func=problem_chain.run,
        description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")
    
]


agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


while True:
    inp = input(":")
    if inp == 'bye':
        break
    print(agent.invoke(
        {"input": inp}
    )['output'])
