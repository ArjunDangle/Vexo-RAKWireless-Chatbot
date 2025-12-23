import os
import logging
import hashlib
import json
import frontmatter # pip install python-frontmatter
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Resilience and AI
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Splitting Logic
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Custom project modules
from etl.schemas import KnowledgeChunk, ProductFamily
from etl.preprocessor import MarkdownPreprocessor

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(filename='mining.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/devstral-2512:free"

# Sequential order for processing families
ORDERED_FAMILIES = ["wisgate", "wisgateos", "wisduo", "wisblock", "software-apis-and-libraries"]

TARGET_FAMILIES = {
    "wisgate": ProductFamily.WISGATE,
    "wisgateos": ProductFamily.WISGATE_OS,
    "wisduo": ProductFamily.WISDUO,
    "wisblock": ProductFamily.WISBLOCK,
    "software-apis-and-libraries": ProductFamily.SOFTWARE
}

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
        3. VERIFICATION: Ensure product ID {product_id} is considered.
        RETURN JSON ONLY. {format_instructions}
        """
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")])
        formatted = prompt.format_messages(
            family=context['family'], 
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
    """Injects table headers into child chunks that contain table rows."""
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

def mine_family(family_key: str, root_dir: str, output_root: str, agent: MiningAgent):
    print(f"\n🚀 --- Starting Family: {family_key.upper()} ---")
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4"), ("#####", "H5"), ("######", "H6")
    ])
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    family_path = os.path.join(root_dir, family_key)
    if not os.path.exists(family_path):
        print(f"⚠️ Path not found: {family_path}")
        return

    for dirpath, _, filenames in os.walk(family_path):
        path_obj = Path(dirpath)
        parts = path_obj.parts
        if "product-categories" not in parts: continue
        idx = parts.index("product-categories")
        if len(parts) <= idx + 2: continue
        
        product_id = parts[idx + 2]
        product_dir = os.path.join(output_root, product_id)
        os.makedirs(product_dir, exist_ok=True)

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
                    context = {"family": family_key, "product_id": product_id, "file_title": file_title}
                    
                    parent_meta = agent.classify_parent(parent.page_content, context)
                    if not parent_meta: continue

                    child_chunks = child_splitter.split_text(parent.page_content)
                    for i, child_text in enumerate(child_chunks):
                        final_content = make_table_aware(parent.page_content, child_text)
                        
                        chunk = parent_meta.model_copy()
                        chunk.content = final_content
                        chunk.product_family = TARGET_FAMILIES[family_key]
                        chunk.product_id = product_id
                        chunk.source_file = file
                        chunk.title = f"{file_title} > {header_path} > {parent_meta.title}"
                        chunk.id = hashlib.md5(f"{file}-{chunk.title}-{i}".encode()).hexdigest()

                        output_file = os.path.join(product_dir, f"{file.replace('.md', '.jsonl')}")
                        with open(output_file, 'a', encoding='utf-8') as out_f:
                            data = chunk.model_dump()
                            data['parent_content'] = parent.page_content
                            out_f.write(json.dumps(data) + "\n")
                print(f"  ✅ {file} processed for {product_id}")
            except Exception as e:
                print(f"  ❌ Error on {file}: {e}")

if __name__ == "__main__":
    agent = MiningAgent()
    INPUT_ROOT = "./data/product-categories" 
    OUTPUT_ROOT = "./storage/mined_knowledge"
    
    for family in ORDERED_FAMILIES:
        mine_family(family, INPUT_ROOT, OUTPUT_ROOT, agent)
    print("\n🎉 Elite Sequential Mining Complete!")