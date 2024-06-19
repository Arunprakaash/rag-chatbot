import os

import chainlit as cl
from chainlit import Message
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from openai_utils import OpenAIUtils
from pinecone_utils import PineconeUtils

load_dotenv()


def get_pinecone_client():
    try:
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD"), region=os.getenv("PINECONE_REGION"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
    except KeyError as e:
        raise KeyError(f"Environment variable {e.args[0]} not set") from None
    return pinecone_client, spec, index_name


def get_openai_client():
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except KeyError as e:
        raise KeyError(f"Environment variable {e.args[0]} not set") from None
    return openai_client


@cl.on_chat_start
async def initialize():
    pinecone_client, spec, index_name = get_pinecone_client()
    pinecone_utils = PineconeUtils(pinecone_client=pinecone_client, pinecone_index_name=index_name)
    openai_utils = OpenAIUtils(openai_client=get_openai_client())

    cl.user_session.set("pinecone_utils", pinecone_utils)
    cl.user_session.set("openai_utils", openai_utils)
    await Message(content="Viamagus AI Initialized!").send()


@cl.on_message
async def main(message: cl.Message):
    pinecone_utils = cl.user_session.get("pinecone_utils")
    openai_utils = cl.user_session.get("openai_utils")
    query_embedding = openai_utils.get_embeddings(message.content)
    relevant_context = pinecone_utils.retrieve_relevant_history(query_embedding)
    response = openai_utils.chat_with_openai(
        query=message.content,
        system_prompt=openai_utils.prepare_system_prompt(context=relevant_context)
    )
    await cl.Message(content=response).send()
