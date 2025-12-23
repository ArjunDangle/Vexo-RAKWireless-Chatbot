import os
import logging
import hashlib
import json
import frontmatter 
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from etl.schemas import KnowledgeChunk, ProductFamily
from etl.preprocessor import MarkdownPreprocessor

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(filename='mining_wisgateos_targeted.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/devstral-2512:free"

# Targeting only WisGateOS variants
TARGET_SUBFOLDERS = ["wisgateos", "wisgateos2"]
PARENT_CATEGORY = "software-apis-and-libraries"

class MiningAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0
        )
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeChunk)
        
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
    def classify_parent(self, text: str, context: dict) -> Optional[KnowledgeChunk]:
        system_prompt = """
        You are a Technical Documentation Specialist for RAKwireless.
        Classify this 'Parent' section.
        1. CATEGORY: [Concept, How-To, Reference, Troubleshooting, Specification].
        2. TITLE: Create a PRECISE title. Include '[TABLE]' if table, '[CODE]' if code.
        3. VERIFICATION: Ensure the specific OS version {product_id} is considered.
        RETURN JSON ONLY. {format_instructions}
        """
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")])
        formatted = prompt.format_messages(
            product_id=context['product_id'],
            text=text[:10000], 
            format_instructions=self.parser.get_format_instructions()
        )
        try:
            response = self.llm.invoke(formatted)
            content = response.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            return self.parser.parse(content)
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            return None

def make_table_aware(parent_text: str, child_text: str) -> str:
    if "|" in child_text:
        lines = parent_text.split("\n")
        headers = []
        for i, line in enumerate(lines):
            if "|" in line and i+1 < len(lines) and "---" in lines[i+1]:
                headers = [lines[i], lines[i+1]]
                break
        if headers and headers[0] not in child_text:
            return f"{headers[0]}\n{headers[1]}\n{child_text}"
    return child_text

def run_targeted_mining():
    agent = MiningAgent()
    root_dir = "./data/product-categories"
    output_root = "./storage/mined_knowledge"
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4"), ("#####", "H5"), ("######", "H6")
    ])
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for subfolder in TARGET_SUBFOLDERS:
        full_path = os.path.join(root_dir, PARENT_CATEGORY, subfolder)
        if not os.path.exists(full_path):
            print(f"⚠️ Skipping missing path: {full_path}")
            continue

        print(f"🚀 Processing: {subfolder.upper()}...")
        product_dir = os.path.join(output_root, subfolder)
        os.makedirs(product_dir, exist_ok=True)

        for dirpath, _, filenames in os.walk(full_path):
            for file in filenames:
                if not file.endswith(".md"): continue
                
                try:
                    with open(os.path.join(dirpath, file), 'r', encoding='utf-8') as f:
                        post = frontmatter.load(f)
                    
                    file_title = post.metadata.get('title', file.replace('.md', ''))
                    _, cleaned_text = MarkdownPreprocessor.process(post.content)
                    parent_docs = markdown_splitter.split_text(cleaned_text)

                    for parent in parent_docs:
                        if len(parent.page_content.strip()) < 50: continue
                        
                        header_path = " > ".join(str(v) for v in parent.metadata.values())
                        context = {"product_id": subfolder, "file_title": file_title}
                        
                        parent_meta = agent.classify_parent(parent.page_content, context)
                        if not parent_meta: continue

                        child_chunks = child_splitter.split_text(parent.page_content)
                        for i, child_text in enumerate(child_chunks):
                            final_content = make_table_aware(parent.page_content, child_text)
                            
                            chunk = parent_meta.model_copy()
                            chunk.content = final_content
                            chunk.product_family = ProductFamily.WISGATE_OS
                            chunk.product_id = subfolder
                            chunk.source_file = file
                            chunk.title = f"{file_title} > {header_path} > {parent_meta.title}"
                            chunk.id = hashlib.md5(f"{file}-{chunk.title}-{i}".encode()).hexdigest()

                            output_file = os.path.join(product_dir, f"{file.replace('.md', '.jsonl')}")
                            with open(output_file, 'a', encoding='utf-8') as out_f:
                                data = chunk.model_dump()
                                data['parent_content'] = parent.page_content
                                out_f.write(json.dumps(data) + "\n")
                    print(f"  ✅ {file} processed")
                except Exception as e:
                    print(f"  ❌ Error on {file}: {e}")

if __name__ == "__main__":
    run_targeted_mining()
    print("\n🎉 Targeted Mining Complete for WisGateOS and WisGateOS 2!")