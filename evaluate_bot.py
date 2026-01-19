import asyncio
import pandas as pd
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # Standard wrapper

from app.services.chat_engine import ChatEngine
from app.core.config import settings

async def run_evaluation():
    # 1. Initialize the Engine
    engine = ChatEngine()
    
    # 2. Setup the "Judge" LLM (OpenRouter)
    # Using a modern wrapper to avoid deprecation warnings
    judge_llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # 3. Setup the Judge Embeddings 
    # Using the standard LangChain wrapper avoids the Pydantic 'string' error
    hf_embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # 4. Prepare Test Data
    test_data = [
        {
            "question": "Does the RAK7289 support LoRa frame filtering?",
            "reference": "Yes, the RAK7289 WisGate Edge Pro supports LoRa Frame Filtering (node whitelisting) in Packet Forwarder mode."
        },
        {
            "question": "What is the default IP address for RAK7268V2?",
            "reference": "The default IP address is 192.168.230.1."
        }
    ]

    results = []
    print(f"🧪 Generating bot responses...")

    for item in test_data:
        # We use the static get_response for evaluation
        response = await engine.get_response(item["question"], history=[])
        results.append({
            "question": item["question"],
            "answer": response["answer"],
            "contexts": [s['title'] for s in response["sources"]],
            "reference": item["reference"]
        })

    dataset = Dataset.from_list(results)

    # 5. Run Evaluation
    print("🧠 Ragas is judging your bot (this may take a minute)...")
    try:
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        # 6. Display results
        df = score.to_pandas()
        print("\n" + "="*60)
        print("📊 RAGAS EVALUATION RESULTS")
        print("="*50)
        
        # Check if results exist before printing
        if not df.empty:
            print(df)
            print(f"\n✅ Average Faithfulness: {df['faithfulness'].mean():.2f}")
            df.to_csv("hallucination_report.csv", index=False)
        else:
            print("⚠️ Evaluation completed but returned no data.")
            
    except Exception as e:
        print(f"❌ Ragas Evaluation Failed: {e}")

if __name__ == "__main__":
    # You might need this for Windows/GitBash environments
    import nest_asyncio
    nest_asyncio.apply()
    
    asyncio.run(run_evaluation())