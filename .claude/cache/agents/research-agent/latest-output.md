# Research Report: Vector Database Options for AI Artist Semantic Search
Generated: 2026-01-31

## Executive Summary

For the ai-artist FastAPI application deployed on Railway with SQLite, **pgvector is the recommended choice**. Railway offers one-click pgvector deployment templates, it provides a natural migration path from SQLite with Alembic, and costs $5-10/month for small instances while keeping vector and relational data unified.

## Research Question

What vector database should be used for semantic artwork search in a Python/FastAPI application currently using SQLite, deployed on Railway, with need for image embedding search?

## Key Findings

### Finding 1: pgvector - Recommended for This Use Case

**Pros:**
- Native PostgreSQL extension - vectors and relational data in one system, one transaction
- Railway has **one-click deployment templates** (pgvector-pg17, pgvector-pg18, postgres-with-pgvector-engine)
- Natural migration from SQLite using existing Alembic setup already in the codebase
- Cost: $5-10/month for small Railway PostgreSQL instance
- Supports up to 16,000 dimensions (OpenAI ada-002 uses 1536, CLIP uses 768)
- Performance: 471 QPS at 99% recall on 50M vectors with pgvectorscale
- Realistic limit: 10-100 million vectors before performance degradation

**Cons:**
- Not as fast as purpose-built vector DBs at billion-scale
- Requires self-management of indexes (HNSW, IVFFlat)
- No built-in embedding generation (need external service)

**Integration with existing codebase:**
- `/Volumes/LizsDisk/ai-artist/src/ai_artist/db/session.py` already has SQLAlchemy setup
- `/Volumes/LizsDisk/ai-artist/alembic/` already configured for migrations
- Change from `sqlite:///` to `postgresql://` connection string
- Add `Vector` column type via pgvector extension

- Source: [Railway pgvector Deploy](https://railway.com/deploy/pgvector-latest), [Timescale pgvector Guide](https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector)

### Finding 2: Chroma - Best for Prototyping Only

**Pros:**
- Lightweight, in-memory by default
- `pip install chromadb` - zero infrastructure
- Built-in Sentence Transformers embeddings
- FastAPI integration is trivial
- ~20ms median search latency on 100k vectors (384 dimensions)

**Cons:**
- **Not production-ready for Railway deployment** - designed for local/embedded use
- No managed hosting option
- SQLite-backed internally (ironic given current stack)
- Memory-bound - doesn't scale well
- No multi-user support

**Verdict:** Good for local prototyping, not for production Railway deployment.

- Source: [Chroma GitHub](https://github.com/chroma-core/chroma), [DataCamp Chroma Tutorial](https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide)

### Finding 3: Pinecone - Overkill for This Scale

**Pros:**
- Fully managed, zero ops
- 50,000 insertions/sec, 5,000 QPS on 1M vectors
- p50 < 10ms, p99 < 50ms at billion-scale
- SOC 2 Type II, HIPAA, GDPR compliant
- Scale to zero (serverless) option

**Cons:**
- **Cost escalates quickly**: $70-2000/month for moderate workloads
- External dependency - network latency added
- No relational data - must maintain separate PostgreSQL
- Two databases to manage instead of one
- Vendor lock-in concerns

**Verdict:** Makes sense at 50M+ vectors or when you need managed enterprise features. Overkill for an artwork gallery with likely <100k images.

- Source: [Aloa Pinecone vs Weaviate](https://aloa.co/ai/comparisons/vector-database-comparison/pinecone-vs-weaviate), [DrCodes Vector DB Guide](https://drcodes.com/posts/pinecone-vs-weaviate-vs-qdrant-2025-vector-database-guide)

### Finding 4: Weaviate - Strong But Complex

**Pros:**
- Open source, can self-host or use managed
- Built-in CLIP modules for image embeddings
- Multi-modal search (text, image, video) out of the box
- GraphQL API, hybrid search
- ~$85/month for 10M vectors (managed)

**Cons:**
- More complex to set up than pgvector
- Separate infrastructure from main database
- Self-hosting: ~$660/month when including DevOps time
- No one-click Railway template

**Verdict:** Best for media companies needing multi-modal search at scale. More infrastructure than needed for this project.

- Source: [Xenoss Vector DB Comparison](https://xenoss.io/blog/vector-database-comparison-pinecone-qdrant-weaviate), [TensorBlue Comparison](https://tensorblue.com/blog/vector-database-comparison-pinecone-weaviate-qdrant-milvus-2025)

## Codebase Analysis

Current database setup in `/Volumes/LizsDisk/ai-artist/src/ai_artist/db/`:

1. **models.py**: `GeneratedImage` model with fields for prompt, scores, tags - perfect for adding an `embedding` column
2. **session.py**: SQLite-specific with WAL mode, needs PostgreSQL adaptation
3. **alembic/**: Already configured for migrations

Current search in `/Volumes/LizsDisk/ai-artist/src/ai_artist/web/helpers.py`:
- `filter_by_search()` does naive string matching on prompts
- Loads JSON metadata files for each image
- No semantic understanding - "sunset" won't find "orange sky over water"

Image count estimate: Gallery structure suggests hundreds to low thousands of artworks, well within pgvector's sweet spot.

## Migration Path: SQLite to pgvector

### Step 1: Deploy PostgreSQL with pgvector on Railway
```bash
# Railway CLI or one-click template
railway add postgresql-with-pgvector
```

### Step 2: Update environment variables
```env
DATABASE_URL=postgresql://...  # Railway provides this
```

### Step 3: Modify session.py for PostgreSQL
```python
# Remove SQLite-specific PRAGMAs
# Use postgresql:// connection string
# Enable pgvector extension via migration
```

### Step 4: Alembic migration to add vector column
```python
# alembic/versions/xxxx_add_embeddings.py
from pgvector.sqlalchemy import Vector

def upgrade():
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.add_column('generated_images',
        Column('embedding', Vector(768))  # CLIP dimension
    )
    op.create_index('ix_embedding_hnsw', 'generated_images', ['embedding'],
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64})
```

### Step 5: Generate embeddings for existing images
- Use OpenAI CLIP or open-source CLIP model
- Batch process existing gallery images
- Store embeddings in new column

### Step 6: Add semantic search endpoint
```python
@app.get("/api/search")
async def semantic_search(query: str, limit: int = 20):
    # Generate query embedding
    query_embedding = clip_model.encode(query)

    # Vector similarity search
    results = db.query(GeneratedImage)\
        .order_by(GeneratedImage.embedding.cosine_distance(query_embedding))\
        .limit(limit)\
        .all()
    return results
```

## Cost Comparison (Monthly)

| Solution | Infrastructure | Notes |
|----------|---------------|-------|
| **pgvector on Railway** | $5-10 | Recommended - unified database |
| Chroma | $0 | Local only, not deployable |
| Pinecone Serverless | $64+ | Overkill, adds complexity |
| Weaviate Managed | $85+ | More infrastructure than needed |
| Weaviate Self-hosted | $660+ | Includes DevOps time |

## Recommendations

### Primary Recommendation: pgvector on Railway

1. **Deploy pgvector** using Railway's one-click template
2. **Migrate from SQLite** using existing Alembic setup
3. **Add embedding column** to GeneratedImage model
4. **Use CLIP embeddings** (768 dimensions) for image-to-text search
5. **Implement hybrid search**: combine vector similarity with metadata filters

### Embedding Strategy

For artwork search, use **CLIP** (Contrastive Language-Image Pre-Training):
- Unified embedding space for text and images
- Search by description ("vibrant sunset landscape")
- Search by similar image (upload reference)
- 768-dimensional vectors, well within pgvector limits

### Index Recommendations

```sql
-- HNSW index for fast approximate search
CREATE INDEX ON generated_images
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Open Questions

1. **Embedding service**: Use OpenAI embeddings API ($0.0001/1K tokens) or self-hosted CLIP model?
   - Self-hosted CLIP is free but requires GPU or slow CPU inference
   - OpenAI is cheap but adds external dependency

2. **Hybrid search weighting**: How to balance semantic similarity with existing scores (aesthetic_score, clip_score)?
   - Suggest: weighted combination in SQL query

3. **Backfill strategy**: Generate embeddings on-demand (lazy) or batch process existing gallery?
   - Recommend: batch process for better UX, then generate on save for new images

4. **Image vs text embeddings**: Store image embeddings, text embeddings (from prompts), or both?
   - Both enables "find similar images" AND "search by description"

## Sources

- [Railway pgvector Deploy Template](https://railway.com/deploy/pgvector-latest)
- [Railway pgvector Hosting Guide](https://blog.railway.com/p/hosting-postgres-with-pgvector)
- [Timescale pgvector Tutorial](https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector)
- [Vector Database Comparison 2025](https://aloa.co/ai/comparisons/vector-database-comparison/pinecone-vs-weaviate-vs-chroma)
- [Pinecone vs Weaviate Guide](https://aloa.co/ai/comparisons/vector-database-comparison/pinecone-vs-weaviate)
- [Chroma GitHub Repository](https://github.com/chroma-core/chroma)
- [FastAPI pgvector RAG Backend](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)
- [Supabase pgvector Docs](https://supabase.com/docs/guides/database/extensions/pgvector)
