# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Environment Setup
- `bun install` - Install dependencies (requires Bun 1.3.11+)
- `bun run profile:init` - Interactive setup to bootstrap a provider profile

### Build & Run
- `bun run build` - Build the project to `dist/`
- `bun run dev` - Build and run the CLI locally
- `bun run start` - Run the built CLI (`node dist/cli.mjs`)
- `bun run smoke` - Quick startup sanity check
- `bun run doctor:runtime` - Validate provider environment variables and reachability

### Testing & Quality
- `bun run typecheck` - Run TypeScript type checking
- `bun run test:provider` - Run tests for API shims and providers
- `bun run hardening:strict` - Full check (typecheck + smoke + doctor)
- `node --test --experimental-strip-types <file>` - Run a single Node.js test file

## High-Level Architecture

OpenClaude is a fork of Claude Code designed to support multiple LLM providers (OpenAI, Gemini, DeepSeek, Ollama, etc.) by shiming the Anthropic SDK.

### API Provider System (`src/services/api/`)
- **`client.ts`**: The central factory for API clients. It routes requests to various shims based on environment variables (e.g., `CLAUDE_CODE_USE_OPENAI`, `CLAUDE_CODE_USE_GEMINI`).
- **`openaiShim.ts`**: Translates Anthropic SDK calls into OpenAI-compatible Chat Completion requests. It handles message format conversion, tool/function calling mapping, and converts OpenAI SSE streams back into Anthropic's stream format.
- **`geminiShim.ts`**: Provides native support for Google's Gemini API.
- **`codexShim.ts`**: Interfaces with the ChatGPT Codex backend.
- **`providerConfig.ts`**: Manages endpoint resolution and model mapping for different providers.

### Core Structure
- **`src/entrypoints/`**: Contains CLI (`cli.tsx`) and main entry points.
- **`src/components/`**: UI components built with React and **Ink** for terminal rendering.
- **`src/tools/`**: Implementation of Claude Code's tools (bash, read, edit, etc.).
- **`src/utils/model/`**: Model-specific configurations, token limits, and provider definitions.
- **`scripts/`**: Development utilities for provider discovery, recommendation, and system checks.

### Key Implementation Patterns
- **Duck-Typing the SDK**: Shims return objects that match the Anthropic SDK interface so the rest of the application remains provider-agnostic.
- **Environment Driven**: Configuration is heavily driven by environment variables (`OPENAI_MODEL`, `OPENAI_BASE_URL`, etc.) to allow switching models without code changes.
