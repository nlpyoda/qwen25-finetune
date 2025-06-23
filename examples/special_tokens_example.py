"""
Example: Working with Special Tokens

This example demonstrates how to add and manage special tokens
for different use cases such as conversation systems, function calling,
and domain-specific applications.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from data.tokenizers.special_tokens import SpecialTokenManager
from models.factory import model_factory

def conversation_tokens_example():
    """
    Example: Adding conversation tokens for chat-based models.
    
    This shows how to add tokens for multi-turn conversations
    with different roles (system, user, assistant).
    """
    print("🗣️  Conversation Tokens Example")
    print("=" * 50)
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print(f"Original vocab size: {len(tokenizer)}")
    
    # Create token manager
    token_manager = SpecialTokenManager(tokenizer)
    
    # Add conversation tokens
    conversation_token_ids = token_manager.add_conversation_tokens()
    print(f"Added {len(conversation_token_ids)} conversation tokens")
    
    # Show added tokens
    conv_tokens = token_manager.list_tokens_by_type(is_system_token=True)
    conv_tokens.extend(token_manager.list_tokens_by_type(is_user_token=True))
    
    print(f"\nConversation tokens:")
    for token in conv_tokens:
        print(f"  {token.token} (ID: {token.token_id}) - {token.description}")
    
    # Example usage in conversation formatting
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
    
    # Format conversation with special tokens
    formatted_text = ""
    for turn in conversation:
        role = turn["role"].upper()
        content = turn["content"]
        formatted_text += f"<{role}>{content}<EOT>"
    
    print(f"\nFormatted conversation:")
    print(f"'{formatted_text}'")
    
    # Tokenize the formatted conversation
    tokens = tokenizer.encode(formatted_text)
    print(f"\nTokenized length: {len(tokens)} tokens")
    
    # Decode to verify
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")
    
    return token_manager

def function_calling_example():
    """
    Example: Adding tokens for function/tool calling.
    
    This demonstrates how to add tokens for structured function calls
    and results, useful for tool-using language models.
    """
    print("\n🔧 Function Calling Tokens Example")
    print("=" * 50)
    
    # Load tokenizer and create token manager
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    token_manager = SpecialTokenManager(tokenizer)
    
    # Add function calling tokens
    function_names = ["search_web", "calculate", "send_email", "get_weather"]
    function_token_ids = token_manager.add_function_tokens(function_names)
    
    print(f"Added {len(function_token_ids)} function-related tokens")
    
    # Show function tokens
    func_tokens = token_manager.list_tokens_by_type(is_function_token=True)
    print(f"\nFunction tokens:")
    for token in func_tokens:
        print(f"  {token.token} (ID: {token.token_id}) - {token.description}")
    
    # Example function call formatting
    function_call_example = """
User: What's the weather like in New York today?