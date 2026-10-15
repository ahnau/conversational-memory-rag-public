# Persistent Conversational Memory — RAG Reference Architecture

A reference implementation illustrating how an external retrieval system can provide continuity across otherwise independent ChatGPT conversations.

## Architecture

The system combines:

- ChatGPT with Project-level instructions
- An MCP interface for retrieval
- A Python retrieval service
- SQLite for structured facts, full-text search (FTS5), and vector retrieval
- Source transcripts, summaries, and atomic memory facts
- Retrieval and handover procedures for maintaining continuity

The general flow is:

**ChatGPT → MCP → Retrieval service → SQLite → Ranked evidence → ChatGPT**

## Repository contents

This repository contains sanitised reference materials illustrating the core architecture:

- **Python retrieval service:** Search, ranking, and MCP-facing retrieval logic.
- **Python ingestion script:** Loading and processing source material.
- **SQL schema:** Core database structures.
- **Project instructions:** Example guidance for retrieving and using external memory.
- **Operating procedure:** Example approach to retrieval, verification, and handover.

## Important notes

These files are architectural references, not a turnkey application. They require adaptation, dependencies, configuration, and security review before deployment.

The repository contains no conversation transcripts, personal memory database, credentials, or live service configuration.

## Discussion

Prepared as supporting material for a discussion on the OpenAI Developer Community forum.
