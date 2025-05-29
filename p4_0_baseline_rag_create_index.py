# create_index.py  – builds a BM25 + Vector hybrid index
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile,
    VectorSearchAlgorithmConfiguration, VectorSearchAlgorithmKind,
    HnswParameters,
)

# 1️⃣  load .env
load_dotenv()
ENDPOINT   = os.getenv("AZURE_SEARCH_ENDPOINT")
ADMIN_KEY  = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "helm-hybrid")
if not ENDPOINT or not ADMIN_KEY:
    raise RuntimeError("Search endpoint or admin key missing in .env")

# 2️⃣  vector section ─ algorithms + profile
algo_cfg = VectorSearchAlgorithmConfiguration(
    name="hnsw-cosine",
    kind=VectorSearchAlgorithmKind.HNSW,        # <- **enum**, not plain string
    parameters=HnswParameters(metric="cosine")  # optional (m, ef*, metric)
)

vector_search = VectorSearch(
    algorithms=[algo_cfg],
    profiles=[VectorSearchProfile(
        name="v1",
        algorithm_configuration_name="hnsw-cosine"
    )]
)

# 3️⃣  searchable + vector fields
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String,
                key=True, filterable=True),
    SearchableField(name="content", type=SearchFieldDataType.String,
                    searchable=True),                           # BM25 text
    SimpleField(name="component", type=SearchFieldDataType.String,
                filterable=True, facetable=True),
    SimpleField(name="env", type=SearchFieldDataType.String,
                filterable=True, facetable=True),
    SimpleField(name="kind", type=SearchFieldDataType.String,
                filterable=True),
    SearchField(                                           # vector column
        name="contentVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,     # text-embedding-3-small
        vector_search_profile_name="v1"
    ),
]

index = SearchIndex(name=INDEX_NAME,
                    fields=fields,
                    vector_search=vector_search)

# 4️⃣  push the index
client = SearchIndexClient(ENDPOINT, AzureKeyCredential(ADMIN_KEY))
print(f"Creating / updating index “{INDEX_NAME}”…")
client.create_or_update_index(index)
print("✅  Index ready")
