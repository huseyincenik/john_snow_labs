"""
LLM service for RAG QA Chatbot Application
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument

from ..config import config, DEFAULT_PROMPT_TEMPLATE
from ..models import ModelProvider, QueryResult, DocumentChunk
from ..utils import app_logger, measure_execution_time, retry_on_exception


class LLMService:
    """Language Model Service for RAG operations"""

    def __init__(self):
        self.logger = app_logger
        self.llm = None
        self.current_provider = None
        self.conversation_chain = None
        self.qa_chain = None
        self.memory = None

    @measure_execution_time
    def initialize_llm(self, model_provider: str, api_key: str) -> None:
        """
        Initialize LLM based on provider

        Args:
            model_provider: Model provider name
            api_key: API key for the provider
        """
        try:
            if model_provider == "OpenAI (API)":
                self.llm = ChatOpenAI(
                    model=config.model.openai_model,
                    temperature=config.model.openai_temperature,
                    max_tokens=config.model.openai_max_tokens,
                    openai_api_key=api_key
                )

                # Initialize memory for conversation
                self.memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True
                )

            elif model_provider == "Local LLM (Qwen)":
                self.llm = ChatOpenAI(
                    model=config.model.local_model,
                    temperature=config.model.local_temperature,
                    max_tokens=config.model.local_max_tokens,
                    openai_api_key='ollama',
                    openai_api_base=config.model.ollama_base_url,
                    request_timeout=120  # 2 minutes timeout for Ollama
                )

                # Initialize memory for conversation
                self.memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True
                )

            else:
                raise ValueError(
                    f"Unsupported model provider: {model_provider}")

            self.current_provider = model_provider
            self.logger.info(f"Initialized LLM for {model_provider}")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    @measure_execution_time
    @retry_on_exception(max_retries=3)
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Tuple[str, Dict, float]],
        conversation_history: Optional[List[Dict]] = None
    ) -> QueryResult:
        """
        Generate answer using RAG approach

        Args:
            query: User question
            retrieved_docs: Retrieved documents from vector store
            conversation_history: Previous conversation messages

        Returns:
            QueryResult object with answer and metadata
        """
        if not self.llm:
            raise ValueError("LLM not initialized. Call initialize_llm first.")

        start_time = time.time()

        try:
            if self.current_provider == "OpenAI (API)":
                result = self._generate_openai_answer(
                    query, retrieved_docs, conversation_history)
            elif self.current_provider == "Local LLM (Qwen)":
                # Use the same OpenAI method since it's using ChatOpenAI interface
                result = self._generate_openai_answer(
                    query, retrieved_docs, conversation_history)
            else:
                raise ValueError(
                    f"Unsupported provider: {self.current_provider}")

            end_time = time.time()
            response_time = end_time - start_time

            # Create DocumentChunk objects from retrieved docs
            source_chunks = []
            for content, metadata, score in retrieved_docs:
                chunk = DocumentChunk(
                    id=metadata.get('chunk_id', ''),
                    document_id=metadata.get('document_id', ''),
                    content=content,
                    chunk_index=metadata.get('chunk_index', 0),
                    start_char=metadata.get('start_char', 0),
                    end_char=metadata.get('end_char', 0),
                    metadata={**metadata, 'similarity_score': score}
                )
                source_chunks.append(chunk)

            query_result = QueryResult(
                query=query,
                answer=result['answer'],
                source_documents=source_chunks,
                model_provider=ModelProvider(self.current_provider),
                response_time=response_time,
                tokens_used=result.get('tokens_used'),
                confidence_score=result.get('confidence_score'),
                metadata={
                    'retrieved_docs_count': len(retrieved_docs),
                    'provider_specific': result.get('metadata', {})
                }
            )

            self.logger.info(
                f"Generated answer in {response_time:.2f}s using {self.current_provider}")
            return query_result

        except Exception as e:
            self.logger.error(f"Failed to generate answer: {str(e)}")
            raise

    def _generate_openai_answer(
        self,
        query: str,
        retrieved_docs: List[Tuple[str, Dict, float]],
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate answer using OpenAI model with conversation chain"""
        try:
            # Create Langchain documents from retrieved docs
            docs = []
            for content, metadata, score in retrieved_docs:
                doc = LangchainDocument(
                    page_content=content,
                    metadata={**metadata, 'similarity_score': score}
                )
                docs.append(doc)

            # Create a simple vector store-like retriever
            class SimpleRetriever:
                def __init__(self, docs):
                    self.docs = docs

                def get_relevant_documents(self, query):
                    return self.docs

            retriever = SimpleRetriever(docs)

            # Create conversation chain if not exists
            if not self.conversation_chain:
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True
                )

            # Generate response
            response = self.conversation_chain({'question': query})

            return {
                'answer': response['answer'],
                'tokens_used': None,  # OpenAI doesn't always provide token count
                'confidence_score': None,
                'metadata': {
                    'source_documents': len(response.get('source_documents', [])),
                    'chat_history_length': len(response.get('chat_history', []))
                }
            }

        except Exception as e:
            self.logger.error(f"OpenAI answer generation failed: {str(e)}")
            raise

    def create_custom_prompt(self, system_message: str = None) -> PromptTemplate:
        """
        Create custom prompt template

        Args:
            system_message: Custom system message

        Returns:
            PromptTemplate object
        """
        if system_message:
            template = f"""
            {system_message}
            
            Context:
            {{context}}
            
            Question: 
            {{question}}
            
            Answer:
            """
        else:
            template = DEFAULT_PROMPT_TEMPLATE

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def update_conversation_memory(self, query: str, answer: str) -> None:
        """
        Update conversation memory with new query and answer

        Args:
            query: User query
            answer: Generated answer
        """
        if self.memory and self.current_provider == "OpenAI":
            try:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(answer)
                self.logger.debug("Updated conversation memory")
            except Exception as e:
                self.logger.warning(f"Failed to update memory: {str(e)}")

    def clear_conversation_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            self.logger.info("Cleared conversation memory")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get current conversation history

        Returns:
            List of conversation messages
        """
        if not self.memory:
            return []

        try:
            messages = []
            for message in self.memory.chat_memory.messages:
                messages.append({
                    'role': message.__class__.__name__.lower().replace('message', ''),
                    'content': message.content
                })
            return messages
        except Exception as e:
            self.logger.warning(
                f"Failed to get conversation history: {str(e)}")
            return []

    def validate_api_connection(self) -> bool:
        """
        Validate API connection by making a simple test call

        Returns:
            True if connection is valid, False otherwise
        """
        if not self.llm:
            return False

        try:
            # For Ollama, only check if the service is reachable
            # Skip LLM test call as it can be very slow for first load
            if self.current_provider == "Local LLM (Qwen)":
                import requests
                try:
                    # Check Ollama health endpoint
                    ollama_url = config.model.ollama_base_url.replace(
                        '/v1', '')
                    health_response = requests.get(
                        f"{ollama_url}/api/tags", timeout=5)
                    if health_response.status_code != 200:
                        self.logger.error(
                            f"Ollama service not ready: {health_response.status_code}")
                        return False

                    # Check if qwen2.5:7b model is available
                    models = health_response.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    if not any('qwen2.5:7b' in name for name in model_names):
                        self.logger.error(
                            "Qwen2.5:7b model not found. Still downloading...")
                        return False

                    self.logger.info(
                        "✅ Ollama service and qwen2.5:7b model are ready!")
                    return True  # Return immediately for Ollama

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Cannot reach Ollama service: {str(e)}")
                    return False

            # For OpenAI, test with a simple prompt
            test_prompt = "Hi"

            response = self.llm.predict(test_prompt)

            self.logger.debug(f"API validation successful: {response[:50]}...")
            return True

        except Exception as e:
            self.logger.error(f"API validation failed: {str(e)}")
            return False


__all__ = ["LLMService"]
