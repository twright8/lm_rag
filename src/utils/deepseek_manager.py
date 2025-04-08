"""
DeepSeek API manager for handling API calls and streaming responses.
"""
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Type

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DeepSeekManager:
    """Manager for DeepSeek API interactions."""

    def __init__(self, config):
        """Initialize the DeepSeek manager with configuration."""
        self.api_key = config["deepseek"]["api_key"]
        self.api_url = config["deepseek"]["api_url"]
        self.use_reasoner = config["deepseek"]["use_reasoner"]
        self.model_name = config["deepseek"]["reasoning_model" if self.use_reasoner else "chat_model"]
        self.temperature = config["deepseek"]["temperature"]
        self.max_tokens = config["deepseek"]["max_tokens"]

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url
            )
            logger.info(f"DeepSeek manager initialized with model: {self.model_name}")
        except ImportError:
            logger.error("Failed to import OpenAI client. Please install with: pip install openai")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing DeepSeek manager: {e}")
            self.client = None

    def generate(self, prompt, system_prompt=None, stream_callback=None):
        """Generate text using DeepSeek API with streaming support for reasoning."""
        if not self.client:
            logger.error("DeepSeek API client not available")
            return "DeepSeek API not available. Please check configuration."

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"Generating with DeepSeek API using model: {self.model_name}")
        logger.info(f"Using reasoner: {self.use_reasoner}")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt[:200]}...")  # Log truncated prompt

        try:
            if stream_callback:
                # Streaming generation
                answer = ""
                reasoning = ""

                logger.info("Starting streaming generation")
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )

                for chunk in stream:
                    # Handle content tokens (the actual response)
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        token = chunk.choices[0].delta.content
                        answer += token
                        logger.debug(f"Response token: {token}")
                        stream_callback(token)

                    # Handle reasoning tokens (only for reasoner model)
                    if self.use_reasoner and chunk.choices and chunk.choices[0].delta.reasoning_content is not None:
                        reasoning_token = chunk.choices[0].delta.reasoning_content
                        reasoning += reasoning_token
                        logger.debug(f"Reasoning token: {reasoning_token}")
                        stream_callback({"type": "thinking", "content": reasoning_token})

                logger.info(f"Streaming generation complete. Response length: {len(answer)}")
                if reasoning:
                    logger.info(f"Reasoning process length: {len(reasoning)}")
                    logger.debug(f"Complete reasoning process: {reasoning}")

                return answer
            else:
                # Non-streaming generation
                logger.info("Starting non-streaming generation")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                # Get content (always available)
                content = response.choices[0].message.content

                # Get reasoning content if available (only for reasoner model)
                reasoning_content = ""
                if self.use_reasoner:
                    reasoning_content = response.choices[0].message.reasoning_content
                    if reasoning_content:
                        logger.info(f"Extracted reasoning content (length: {len(reasoning_content)})")
                        logger.debug(f"Reasoning content: {reasoning_content}")

                logger.info(f"Non-streaming generation complete. Response length: {len(content)}")
                logger.debug(f"Response: {content[:200]}...")

                # Return both content and reasoning
                return {
                    "content": content.strip(),
                    "reasoning": reasoning_content
                }

        except Exception as e:
            logger.error(f"Error generating with DeepSeek API: {e}", exc_info=True)
            return f"Error: {str(e)}"