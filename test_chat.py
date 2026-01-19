import requests
import json

API_URL = "http://127.0.0.1:8000/chat/static"

def chat_session():
    print("\n" + "="*50)
    print("🤖 RAK Knowledge Engine")
    print("="*50)
    chat_history = []

    while True:
        user_query = input("\n👉 You: ").strip()
        if user_query.lower() in ['exit', 'quit']: break
        
        payload = {"message": user_query, "history": chat_history}
        
        try:
            response = requests.post(API_URL, json=payload)
            data = response.json()
            
            print(f"🤖 AI ({data['latency']:.2f}s):")
            print(data['answer'])
            
            if data.get('sources'):
                print("\n   📚 Sources Used:")
                for idx, s in enumerate(data['sources'], 1):
                    print(f"      {idx}. {s['title']} [Conf: {s['confidence']:.2f}]")
            
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": data['answer']})
            if len(chat_history) > 10: chat_history = chat_history[-10:]

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat_session()