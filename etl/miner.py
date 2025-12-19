import os
import logging
import hashlib
import argparse
import frontmatter  # pip install python-frontmatter
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 1. Resilience
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 2. LangChain & Text Splitters
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 3. Our Custom Modules
from etl.schemas import KnowledgeChunk, ProductFamily
from etl.preprocessor import MarkdownPreprocessor

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(
    filename='mining.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/devstral-2512:free"

TARGET_FAMILIES = {
    "wisgate": ProductFamily.WISGATE,
    "wisgateos": ProductFamily.WISGATE_OS,
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
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception))
    )
    def classify_chunk(self, text: str, context: dict) -> Optional[KnowledgeChunk]:
        """
        Uses LLM to classify the chunk category.
        """
        system_prompt = """
        You are a Technical Documentation Specialist.
        1. Classify the text into one of the 8 Knowledge Categories (Concept, How-To, etc.).
        2. The 'Title' field is pre-filled. You may keep it or slightly refine it for clarity.
        3. Assign the correct Product Family and ID.
        
        Context:
        - Family: {family}
        - Product ID: {product_id}
        - File Description: {file_desc}
        
        RETURN JSON ONLY.
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Title: {title}\n\nContent: {text}")
        ])

        formatted_prompt = prompt.format_messages(
            family=context.get('family'),
            product_id=context.get('product_id'),
            file_desc=context.get('file_description', ''),
            title=context.get('derived_title'),
            text=text[:4000], 
            format_instructions=self.parser.get_format_instructions()
        )

        try:
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            chunk_data = self.parser.parse(content)
            
            # Inject System Data
            chunk_data.content = text
            chunk_data.product_family = context['family_enum']
            chunk_data.product_id = context['product_id']
            chunk_data.source_file = context['filepath']
            
            # Deterministic ID
            unique_str = f"{context['filepath']}-{context['derived_title']}-{text[:50]}"
            chunk_data.id = hashlib.md5(unique_str.encode()).hexdigest()
            
            return chunk_data

        except Exception as e:
            logging.error(f"LLM Error on {context['filepath']}: {e}")
            return None

def mine_directory(root_dir: str, output_dir: str, target_family: str = None):
    try:
        agent = MiningAgent()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🚀 Starting Miner (Frontmatter + Deep Split)...")
    if target_family:
        print(f"🎯 Target Family: {target_family}")

    # 1. LOGICAL SPLITTER (Headers)
    # We split deep to catch specific tables (like Pin Definitions)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 2. PHYSICAL SPLITTER (Safety Net)
    # Ensures no single chunk exceeds embedding model limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    success_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        path_obj = Path(dirpath)
        parts = path_obj.parts
        
        if "product-categories" not in parts: continue
        idx = parts.index("product-categories")
        if len(parts) <= idx + 2: continue
            
        level_1 = parts[idx + 1]
        level_2 = parts[idx + 2]

        current_family_str = level_1
        current_product_id = level_2
        
        if level_2 in TARGET_FAMILIES:
            current_family_str = level_2
            current_family_enum = TARGET_FAMILIES[level_2]
        elif level_1 in TARGET_FAMILIES:
            current_family_str = level_1
            current_family_enum = TARGET_FAMILIES[level_1]
        else:
            continue

        # FILTER LOGIC
        if target_family and (target_family != level_1 and target_family != level_2):
            continue

        for file in filenames:
            if not file.endswith(".md"): continue

            full_path = os.path.join(dirpath, file)
            rel_path = os.path.relpath(full_path, start=root_dir)
            
            try:
                # --- A. Parse Frontmatter ---
                with open(full_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
                    
                raw_content = post.content
                front_meta = post.metadata
                
                # Extract Metadata
                file_title = front_meta.get('title', file.replace('.md', ''))
                file_desc = front_meta.get('description', '')
                
                # --- B. Preprocess Content ---
                _, cleaned_text = MarkdownPreprocessor.process(raw_content)
                
                # --- C. PASS 1: Header Splits ---
                header_splits = markdown_splitter.split_text(cleaned_text)
                
                # --- D. PASS 2: Recursive Splits ---
                final_splits = text_splitter.split_documents(header_splits)
                
                for split in final_splits:
                    if len(split.page_content.strip()) < 30: continue

                    # Construct Breadcrumb Title
                    header_path = " > ".join(str(v) for k, v in split.metadata.items())
                    
                    if not header_path:
                        derived_title = file_title
                    else:
                        derived_title = f"{file_title} > {header_path}"

                    # Context Injection
                    context_aware_text = (
                        f"**Source:** {file_title}\n"
                        f"**Context:** {header_path}\n"
                        f"**Summary:** {file_desc}\n\n"
                        f"{split.page_content}"
                    )

                    context = {
                        "family": current_family_str,
                        "family_enum": current_family_enum,
                        "product_id": current_product_id,
                        "filename": file,
                        "filepath": rel_path,
                        "derived_title": derived_title,
                        "file_description": file_desc
                    }

                    chunk_obj = agent.classify_chunk(context_aware_text, context)
                    
                    if chunk_obj:
                        output_file = os.path.join(output_dir, f"{current_product_id}.jsonl")
                        with open(output_file, 'a', encoding='utf-8') as out_f:
                            out_f.write(chunk_obj.model_dump_json() + "\n")
                
                success_count += 1
                print(f"[{current_product_id}] Processed: {file} ({len(final_splits)} chunks)")

            except Exception as e:
                print(f"❌ Error on {file}: {e}")
                continue 

    print(f"\n🎉 Mining Complete. Files Processed: {success_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=False, help="Filter by family (e.g., wisduo, wisblock)")
    args = parser.parse_args()
    
    INPUT_ROOT = "./data/product-categories" 
    OUTPUT_ROOT = "./storage/mined_knowledge"
    
    if os.path.exists(INPUT_ROOT):
        mine_directory(INPUT_ROOT, OUTPUT_ROOT, target_family=args.family)