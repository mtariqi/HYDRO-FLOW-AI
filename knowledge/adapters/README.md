# Production graph-RAG adapter

The default `LocalEvidenceGraph` is deterministic and works without credentials.
For production, normalize documents and experiment records into the same
`Evidence` schema, then index them through RAG-Anything/LightRAG.

Recommended sequence:

1. Parse PDFs, DOCX, tables, equations, hydrographs, and maps with RAG-Anything.
2. Add station and experiment metadata as structured evidence records.
3. Use LightRAG hybrid retrieval for local entity and global thematic queries.
4. Return evidence IDs, source paths, page/figure references, and run IDs.
5. Verify numerical claims directly against the metrics store before answering.

Raw 15-minute observations should remain in Parquet or a time-series database;
only summaries, events, metadata, and artifact references belong in the graph.

