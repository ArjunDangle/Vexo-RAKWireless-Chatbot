import os
import logging
import hashlib
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

# 1. Resilience Libraries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 2. LangChain & LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

# 3. Our Custom Modules
from etl.schemas import KnowledgeChunk, KnowledgeCategory, ProductFamily
from etl.preprocessor import MarkdownPreprocessor

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(
    filename='mining.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/devstral-2512:free"

# Target Families Mapping
# Matches folder names to our Schema Enum
TARGET_FAMILIES = {
    "wisgate": ProductFamily.WISGATE,
    "wisgateos": ProductFamily.WISGATE_OS,  # This is a folder inside software-apis...
    "wisduo": ProductFamily.WISDUO,
    "wisblock": ProductFamily.WISBLOCK,
    "software-apis-and-libraries": ProductFamily.SOFTWARE
}

class MiningAgent:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is missing from .env file")

        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://rakwireless.com", 
                "X-Title": "RAK Knowledge Engine"
            }
        )
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeChunk)
        
    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((Exception))
    )
    def classify_chunk(self, text: str, context: dict) -> Optional[KnowledgeChunk]:
        """
        Sends text to LLM to extract Category and Metadata.
        """
        system_prompt = """
        You are a Technical Documentation Specialist for RAKwireless.
        Analyze the provided markdown text.
        1. Classify it into one of the 8 Knowledge Categories (Concept, How-To, etc.).
        2. Generate a concise, descriptive Title.
        3. Assign the correct Product Family and ID based on the provided context keys.
        
        Context provided by system:
        - Family: {family}
        - Product ID: {product_id}
        - Filename: {filename}
        
        RETURN ONLY JSON. NO MARKDOWN BLOCK.
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ])

        formatted_prompt = prompt.format_messages(
            family=context.get('family'),
            product_id=context.get('product_id'),
            filename=context.get('filename'),
            text=text[:12000], 
            format_instructions=self.parser.get_format_instructions()
        )

        try:
            response = self.llm.invoke(formatted_prompt)
            
            # Sanitize Output
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            chunk_data = self.parser.parse(content)
            
            # Inject raw content and context back into object
            chunk_data.content = text
            chunk_data.product_family = context['family_enum']
            chunk_data.product_id = context['product_id']
            chunk_data.source_file = context['filepath']
            
            # Generate Deterministic ID
            unique_str = f"{context['filepath']}-{text[:50]}"
            chunk_data.id = hashlib.md5(unique_str.encode()).hexdigest()
            
            return chunk_data

        except OutputParserException as e:
            logging.error(f"JSON Parsing failed for {context['filepath']}: {e}")
            return None
        except Exception as e:
            logging.warning(f"LLM Call failed, retrying... Error: {e}")
            raise e

def mine_directory(root_dir: str, output_dir: str, target_family: str = None):
    try:
        agent = MiningAgent()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🚀 Starting Miner on: {root_dir}")
    print(f"🎯 Target Family: {target_family or 'ALL'}")
    
    success_count = 0
    failure_count = 0

    # Walk the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        path_obj = Path(dirpath)
        parts = path_obj.parts
        
        # 1. Locate the anchor folder "product-categories"
        if "product-categories" not in parts:
            continue
            
        idx = parts.index("product-categories")
        
        # We need at least 2 levels deeper: .../product-categories/<LEVEL_1>/<LEVEL_2>
        if len(parts) <= idx + 2:
            continue
            
        level_1 = parts[idx + 1]  # e.g., "wisgate", "software-apis-and-libraries"
        level_2 = parts[idx + 2]  # e.g., "rak7240", "wisgateos"

        # 2. Determine Family Enum & Product ID
        current_family_str = level_1
        current_product_id = level_2
        
        # Logic: If Level 2 (e.g., wisgateos) is a known target family, use it as the primary family
        if level_2 in TARGET_FAMILIES:
            current_family_str = level_2
            current_family_enum = TARGET_FAMILIES[level_2]
        elif level_1 in TARGET_FAMILIES:
            current_family_str = level_1
            current_family_enum = TARGET_FAMILIES[level_1]
        else:
            # Skip unknown folders
            continue

        # 3. Apply Filter (--family argument)
        if target_family:
            # Check if target matches either Level 1 or Level 2
            # This handles: --family wisgateos matching .../software.../wisgateos
            if target_family != level_1 and target_family != level_2:
                continue

        for file in filenames:
            if not file.endswith(".md"):
                continue

            full_path = os.path.join(dirpath, file)
            # Relative path for cleaner IDs
            rel_path = os.path.relpath(full_path, start=root_dir)
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                meta, cleaned_text = MarkdownPreprocessor.process(raw_content)
                sections = cleaned_text.split("\n## ")
                
                for i, section in enumerate(sections):
                    if len(section.strip()) < 50: continue
                    if i > 0: section = "## " + section

                    context = {
                        "family": current_family_str,
                        "family_enum": current_family_enum,
                        "product_id": current_product_id,
                        "filename": file,
                        "filepath": rel_path # Use relative path for cleaner metadata
                    }

                    chunk_obj = agent.classify_chunk(section, context)
                    
                    if chunk_obj:
                        output_file = os.path.join(output_dir, f"{current_product_id}.jsonl")
                        with open(output_file, 'a', encoding='utf-8') as out_f:
                            out_f.write(chunk_obj.model_dump_json() + "\n")
                
                success_count += 1
                print(f"[{current_product_id}] Processed: {file}")

            except Exception as e:
                failure_count += 1
                print(f"❌ Error on {file}: {e}")
                # Log to file
                log_file = f"{target_family or 'general'}_errors.log"
                with open(log_file, "a") as err_f:
                    err_f.write(f"{full_path} | {str(e)}\n")
                continue 

    print(f"\n🎉 Mining Complete. Success: {success_count} | Failed: {failure_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAK Knowledge Miner")
    parser.add_argument("--family", type=str, help="Specific family to mine (wisgate, wisgateos, wisduo, wisblock)", required=False)
    
    args = parser.parse_args()

    # UPDATED DEFAULT PATH
    INPUT_ROOT = "./data/product-categories" 
    OUTPUT_ROOT = "./storage/mined_knowledge"
    
    if not os.path.exists(INPUT_ROOT):
        print(f"⚠️  WARNING: Input directory '{INPUT_ROOT}' does not exist.")
        print("Please check your folder structure.")
    else:
        mine_directory(INPUT_ROOT, OUTPUT_ROOT, target_family=args.family)