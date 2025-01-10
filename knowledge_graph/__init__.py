import os
import random
from dotenv import load_dotenv
from graphrag_sdk.source import TEXT
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.models.litellm import LiteModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
load_dotenv()

def list_md_files(folder):
    md_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files


# Model - vendor: gemini, model: gemini-2.0-flash-exp -> gemini/gemini-2.0-flash-exp
model = LiteModel(model_name="gemini/gemini-2.0-flash-exp")
print("Model loaded")

# print all paths
paths = list_md_files('docs')
print(paths)

# Define the subset of sources for the Ontology geenration
sources = [TEXT(path) for path in random.sample(paths, 10)]

# Ontology Auto-Detection
ontology = Ontology.from_sources(
    sources=sources,
    model=model,
)

print("Ontology auto-detected")
print(ontology)

kg = KnowledgeGraph(
    name="kg_name",
    model_config=KnowledgeGraphModelConfig.with_model(model),
    ontology=ontology,
    host="127.0.0.1",
    port=6379,
    # username=falkor_username, # optional
    # password=falkor_password  # optional
)

# Load all the files in the docs folder for the Knowledge Graph generation
sources = [TEXT(path) for path in paths]

kg.process_sources(sources)
print("Knowledge Graph loaded")