import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from openai_utils import OpenAIUtils
from pinecone_utils import PineconeUtils

load_dotenv()


def get_pinecone_client():
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        spe = ServerlessSpec(cloud=os.environ["PINECONE_CLOUD"], region=os.environ["PINECONE_REGION"])
        index = os.environ["PINECONE_INDEX_NAME"]
    except KeyError as e:
        raise KeyError(f"Environment variable {e.args[0]} not set") from None
    return pc, spe, index


def get_openai_client():
    try:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except KeyError as e:
        raise KeyError(f"Environment variable {e.args[0]} not set") from None
    return openai_client


pinecone_client, spec, index_name = get_pinecone_client()
pinecone_utils = PineconeUtils(pinecone_client, index_name)
openai_utils = OpenAIUtils(get_openai_client())


def main():
    parser = argparse.ArgumentParser(description='Viamagus AI.')
    parser.add_argument('query', type=str, help='The query to ask Viamagus AI.')
    args = parser.parse_args()

    query_embedding = openai_utils.get_embeddings(args.query)
    relevant_context = pinecone_utils.retrieve_relevant_history(query_embedding)
    response = openai_utils.chat_with_openai(args.query, system_prompt=openai_utils.prepare_system_prompt(
        context=relevant_context))
    print(f"\nFinal Test Prompt: \n{args.query}\n")
    print(f"Context Referred: \n{relevant_context}\n")
    print(f"Final Test Prompt Response: \n{response}\n")


if __name__ == "__main__":
    main()
