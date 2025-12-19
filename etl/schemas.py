from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# 1. The Strict "Core Four" Families
class ProductFamily(str, Enum):
    WISGATE = "wisgate"
    WISGATE_OS = "wisgateos"
    WISDUO = "wisduo"
    WISBLOCK = "wisblock"
    SOFTWARE = "software-apis-and-libraries" # Handling that RUI3 folder
    UNKNOWN = "unknown"

# 2. The 8 Knowledge Categories
class KnowledgeCategory(str, Enum):
    CONCEPT = "Concept"
    HOW_TO = "How-To"
    TROUBLESHOOTING = "Troubleshooting"
    REFERENCE = "Reference"  # Hardware specs, tables
    API = "API"              # Code, commands, JSON configs
    RELEASE = "Release"
    COMPATIBILITY = "Compatibility"
    GENERAL = "General"

# 3. The Atomic Knowledge Chunk
class KnowledgeChunk(BaseModel):
    """
    Represents a single, atomic piece of information extracted from a document.
    """
    id: str = Field(
        ..., 
        description="Deterministic ID: hash(filename + chunk_index). Prevents duplicates."
    )
    product_family: ProductFamily = Field(
        ..., 
        description="The high-level product family."
    )
    product_id: Optional[str] = Field(
        None, 
        description="The specific device ID (e.g., 'rak7240', 'rui3'). Extracted from folder path."
    )
    category: KnowledgeCategory = Field(
        ..., 
        description="The strict classification of this chunk."
    )
    title: str = Field(
        ..., 
        description="A descriptive title (e.g., 'RAK7240 Power Consumption')."
    )
    content: str = Field(
        ..., 
        description="The cleaned markdown text."
    )
    keywords: List[str] = Field(
        default_factory=list, 
        description="Keywords extracted from YAML frontmatter."
    )
    source_file: str = Field(
        ..., 
        description="The relative filename (e.g., 'wisgate/rak7240/datasheet.md')."
    )
    token_count: int = Field(
        0, 
        description="Number of tokens in this chunk (for cost tracking)."
    )