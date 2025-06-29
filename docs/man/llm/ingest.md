---
title: llm ingest
command:
  name: ingest
  usage: ingest [flags]
  description: Ingest OpenTDF documentation for RAG (Retrieval-Augmented Generation)
---

# llm ingest

Ingest OpenTDF documentation into a vector database for Retrieval-Augmented Generation (RAG). This enables the LLM to provide more accurate responses by retrieving relevant documentation context.

## Usage

```shell
otdfctl llm ingest [flags]
```

## Flags

- `--embedding-model` - Path to the embedding model file (defaults to llama3.2:1b)
- `--index-path` - Path to save the vector index (default: ~/.otdfctl/rag_index.json)
- `--source` - Source type: 'github' or 'local' (default: github)
- `--path` - Path to local docs directory (required when --source=local)
- `--cache-dir` - Directory for caching downloaded docs (default: ~/.otdfctl/doc_cache)

## Examples

Ingest from OpenTDF GitHub repository:
```shell
otdfctl llm ingest --source github
```

Ingest from local documentation directory:
```shell
otdfctl llm ingest --source local --path /path/to/docs
```

Use custom embedding model and index path:
```shell
otdfctl llm ingest --embedding-model /path/to/model.gguf --index-path ./my_index.json
```

## Process

1. **Document Download/Reading**: Downloads markdown files from the OpenTDF docs repository or reads from local directory
2. **Text Processing**: Cleans and chunks the documentation into smaller pieces for better retrieval
3. **Embedding Generation**: Creates vector embeddings for each document chunk using the specified model
4. **Index Creation**: Builds a searchable vector index and saves it to disk

## Performance Notes

- Initial ingestion may take 10-30 minutes depending on document count and model speed
- Embeddings are cached and only regenerated when documents change
- Smaller embedding models are faster but may be less accurate for retrieval
- The index file can be shared across systems to avoid re-processing

## Integration

After ingestion, use the `llm chat` command with the `--rag` flag to enable context-aware responses:

```shell
otdfctl llm chat model.gguf --rag
```