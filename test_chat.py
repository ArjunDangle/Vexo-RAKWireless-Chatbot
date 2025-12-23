import requests
import time
import sys

# Configuration
API_URL = "http://127.0.0.1:8000/chat"

def type_writer(text, delay=0.01):
    """Effect to print text like a typewriter"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def chat_session():
    print("\n" + "="*50)
    print("🤖 RAK Knowledge Engine - Terminal Client")
    print("   Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_query = input("\n👉 You: ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\n👋 Exiting. Goodbye!")
                break
            
            if not user_query:
                continue

            print("   Thinking...", end="\r")

            # Send Request to API
            try:
                start_ts = time.time()
                response = requests.post(API_URL, json={"message": user_query}) # Use 'message', not 'query'
                response.raise_for_status()
                data = response.json()
                
                # Calculate total round-trip time
                latency = data.get('latency', 0)
                
                # Clear "Thinking..." line
                print(" " * 20, end="\r")

                # Print Answer
                print(f"🤖 AI ({latency:.2f}s):")
                type_writer(data['answer'])
                
                # Print Sources
                if data.get('sources'):
                    print("\n   📚 Sources Used:")
                    for idx, source in enumerate(data['sources'], 1):
                        print(f"      {idx}. {source['title']} ({source['url']})")
                else:
                    print("\n   ⚠️ No specific sources found.")

            except requests.exceptions.ConnectionError:
                print("\n❌ Error: Could not connect to the API.")
                print("   Is the server running? (uvicorn app.main:app --reload)")
            except Exception as e:
                print(f"\n❌ Error: {e}")

        except KeyboardInterrupt:
            print("\n👋 Exiting. Goodbye!")
            break

if __name__ == "__main__":
    chat_session()