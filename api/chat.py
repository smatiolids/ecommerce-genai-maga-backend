from flask import Blueprint, jsonify, request
import uuid
import os
from app import session, keyspace
from langchain.memory import CassandraChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

bp = Blueprint('chat', __name__)
memory_table_name = 'vs_investment_memory'
kb_table_name = 'vs_investment_kb'

llm = OpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'), temperature=0.1)
embedding_generator = OpenAIEmbeddings(
    openai_api_key=os.environ.get('OPENAI_API_KEY'))

CassVectorStore = Cassandra(
    session=session,
    keyspace=keyspace,
    table_name=kb_table_name,
    embedding=embedding_generator
)

index = VectorStoreIndexWrapper(
    vectorstore=CassVectorStore
)


@bp.route('/api/sensor', methods=['POST'])
def get_data():
    res = session.execute("select * from tlp_stress.sensor_data limit 10")
    data = []
    for row in res:
        data.append({**row})
    return jsonify(data)


@bp.route('/api/chat', methods=['POST'])
def post_chat():
    data = request.get_json()

    if 'conversation_id' not in data:
        data['conversation_id'] = str(uuid.uuid4())

    prompt_template = """
    Given the following extracted parts of a long document and a question, create a final answer in a very short format. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Answer in Portuguese.


    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    message_history = CassandraChatMessageHistory(
        session_id=data['conversation_id'],
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
        table_name=memory_table_name
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=50,
        buffer=""
    )

    retrieverSim = CassVectorStore.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 5,
            # 'filter': {"source": st.session_state.file}
        },
    )

    # Create a "RetrievalQA" chain
    chainSim = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retrieverSim,
        memory=memory,
        chain_type_kwargs={
            'prompt': PROMPT,
            'document_variable_name': 'summaries'
        }
    )
    new_summary = memory.predict_new_summary(
        memory.chat_memory.messages,
        memory.moving_summary_buffer,
    )

    print(data)

    # Run it and print results
    answer = chainSim.run(data['query'])

    return jsonify({'answer': answer, 'summary': new_summary, 'conversation_id': data['conversation_id']})
