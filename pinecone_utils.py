from typing import List, Optional, Union

from pinecone import Pinecone, Vector


class PineconeUtils:
    def __init__(self, pinecone_client: Pinecone, pinecone_index_name: str):

        self.pc_client = pinecone_client
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_index = self.__initialize_pinecone_index()

    def __initialize_pinecone_index(self):
        pinecone_index = None
        try:

            # Check if the index exists
            if self.pinecone_index_name in self.pc_client.list_indexes().names():
                pinecone_index = self.pc_client.Index(self.pinecone_index_name)
            else:
                print(f"Index '{self.pinecone_index_name}' does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Finished initializing Pinecone index.")
            return pinecone_index

    def add_embeddings_to_pinecone(self, embeddings: Union[List[Vector], List[tuple], List[dict]]):
        try:
            if self.pinecone_index is None:
                raise ValueError(f"Index {self.pinecone_index_name} does not exist.")

            self.pinecone_index.upsert(vectors=embeddings)

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Finished processing embeddings.")

    def retrieve_relevant_history(self, query_embedding: float, top_k: Optional[int] = 2):
        try:
            # Initialize Pinecone index
            if self.pinecone_index is None:
                raise ValueError(f"Index {self.pinecone_index_name} does not exist.")

            # Search for the most relevant message in the chat history
            search_results = self.pinecone_index.query(vector=[query_embedding], top_k=top_k,
                                                       include_metadata=True)

            # Get the most relevant message
            relevant_message = '\n'.join([r['metadata']['text'] for r in search_results['matches']])

            return relevant_message

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Finished retrieving relevant history.")
