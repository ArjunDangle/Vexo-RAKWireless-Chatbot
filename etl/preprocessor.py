import re
import yaml
from typing import Tuple, List, Dict

class MarkdownPreprocessor:
    """
    Handles Docusaurus-specific syntax and Frontmatter extraction.
    Complementary to the user's 'data_cleaning.py'.
    """

    @staticmethod
    def extract_frontmatter(content: str) -> Tuple[Dict, str]:
        """
        Splits the YAML frontmatter from the markdown content.
        Returns: ({metadata_dict}, cleaned_content_string)
        """
        # Regex to find --- YAML --- blocks at the start of the file
        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(pattern, content, re.DOTALL)
        
        metadata = {}
        cleaned_content = content

        if match:
            yaml_text = match.group(1)
            try:
                metadata = yaml.safe_load(yaml_text) or {}
                # Remove the frontmatter from the content body
                cleaned_content = content[match.end():]
            except yaml.YAMLError:
                pass # If YAML fails, treat it as text
        
        return metadata, cleaned_content

    @staticmethod
    def clean_admonitions(content: str) -> str:
        """
        Converts Docusaurus :::warning blocks to standard Markdown bold headers.
        Example: :::warning Text ::: -> **WARNING:** Text
        """
        # Map common Docusaurus types to Display Names
        mappings = {
            "warning": "WARNING",
            "tip": "TIP",
            "note": "NOTE",
            "info": "INFO",
            "caution": "CAUTION",
            "danger": "DANGER"
        }

        # Regex to capture :::type content :::
        # This handles single-line and multi-line blocks
        pattern = r":::(\w+)\s*(.*?)\s*:::"
        
        def replace(match):
            dtype = match.group(1).lower()
            text = match.group(2).strip()
            header = mappings.get(dtype, dtype.upper())
            return f"\n> **{header}:** {text}\n"

        # Apply regex with DOTALL to catch multi-line blocks
        return re.sub(pattern, replace, content, flags=re.DOTALL)

    @staticmethod
    def process(content: str) -> Tuple[Dict, str]:
        """
        Main entry point.
        """
        # 1. Extract Metadata
        meta, text = MarkdownPreprocessor.extract_frontmatter(content)
        
        # 2. Convert ::: syntax
        text = MarkdownPreprocessor.clean_admonitions(text)
        
        return meta, text