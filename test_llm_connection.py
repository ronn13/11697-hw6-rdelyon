"""
Test script to verify LLM API connections
Tests both llama and OpenAI connections with a simple question
"""

import configparser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

def test_llama():
    """Test llama connection"""
    print("=" * 60)
    print("Testing llama Connection")
    print("=" * 60)
    
    try:
        print("\n1. Initializing ChatGoogleGenerativeAI...")
        
        llm = ChatOpenAI(
            model="llama3-2-11b-instruct",
            api_key=config.get('openai', 'openai_api_key'),
            base_url=config.get('openai', 'base_url')
        )
        print("   ✓ LLM initialized")
        
        print("\n2. Creating prompt...")
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer this question in one word: {question}"
        )
        print("   ✓ Prompt created")
        
        print("\n3. Creating chain...")
        chain = prompt | llm | StrOutputParser()
        print("   ✓ Chain created")
        
        print("\n4. Making API call with question: 'What is 2+2?'")
        print("   (This may take a few seconds...)")
        start_time = time.time()
        
        try:
            answer = chain.invoke({"question": "What is 2+2?"})
            elapsed = time.time() - start_time
            print(f"   ✓ Response received in {elapsed:.2f} seconds")
            print(f"   Response: {answer}")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ✗ Error after {elapsed:.2f} seconds: {e}")
            print(f"   Error type: {type(e).__name__}")
            return False
            
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")
        return False


def test_openai():
    """Test OpenAI connection"""
    print("\n" + "=" * 60)
    print("Testing OpenAI Connection")
    print("=" * 60)
    
    try:
        print("\n1. Initializing ChatOpenAI...")
        llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            api_key=config.get('openai', 'openai_api_key'),
            base_url=config.get('openai', 'base_url')
        )
        print("   ✓ LLM initialized")
        
        print("\n2. Creating prompt...")
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer this question in one word: {question}"
        )
        print("   ✓ Prompt created")
        
        print("\n3. Creating chain...")
        chain = prompt | llm | StrOutputParser()
        print("   ✓ Chain created")
        
        print("\n4. Making API call with question: 'What is 2+2?'")
        print("   (This may take a few seconds...)")
        start_time = time.time()
        
        try:
            answer = chain.invoke({"question": "What is 2+2?"})
            elapsed = time.time() - start_time
            print(f"   ✓ Response received in {elapsed:.2f} seconds")
            print(f"   Response: {answer}")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ✗ Error after {elapsed:.2f} seconds: {e}")
            print(f"   Error type: {type(e).__name__}")
            return False
            
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")
        return False


def test_direct_api_call():
    """Test direct API call without LangChain chain"""
    print("\n" + "=" * 60)
    print("Testing Direct llama API Call (without chain)")
    print("=" * 60)
    
    try:
        print("\n1. Initializing ChatGoogleGenerativeAI...")
        llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            api_key=config.get('openai', 'openai_api_key'),
            base_url=config.get('openai', 'base_url')
        )
        print("   ✓ LLM initialized")
        
        print("\n2. Making direct invoke call...")
        print("   (This may take a few seconds...)")
        start_time = time.time()
        
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content="Say 'test' and nothing else.")
        
        try:
            response = llm.invoke([message])
            elapsed = time.time() - start_time
            print(f"   ✓ Response received in {elapsed:.2f} seconds")
            print(f"   Response: {response.content}")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ✗ Error after {elapsed:.2f} seconds: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback:")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LLM Connection Test Script")
    print("=" * 60)
    print(f"\nBase URL: {config.get('openai', 'base_url')}")
    print(f"llama API Key: {config.get('openai', 'openai_api_key')[:10]}...")
    print(f"OpenAI API Key: {config.get('openai', 'openai_api_key')[:10]}...")
    
    results = {}
    
    # Test direct API call first (simplest)
    print("\n" + "=" * 60)
    print("TESTING DIRECT API CALL FIRST (Simplest)")
    print("=" * 60)
    results['direct'] = test_direct_api_call()
    
    # Test OpenAI
    results['openai'] = test_openai()
    
    # Test llama
    results['llama'] = test_llama()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Direct API Call: {'✓ PASSED' if results['direct'] else '✗ FAILED'}")
    print(f"llama (with chain): {'✓ PASSED' if results['llama'] else '✗ FAILED'}")
    print(f"OpenAI (with chain): {'✓ PASSED' if results['openai'] else '✗ FAILED'}")
    print("=" * 60)
    
    if not any(results.values()):
        print("\n⚠ All tests failed. Possible issues:")
        print("  - Network connectivity to API gateway")
        print("  - Invalid API keys")
        print("  - API gateway is down or slow")
        print("  - Firewall/proxy blocking requests")


if __name__ == "__main__":
    main()

