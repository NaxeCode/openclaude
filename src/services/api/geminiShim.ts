/**
 * Native Google Gemini API shim for Claude Code.
 *
 * Translates Anthropic SDK calls (anthropic.beta.messages.create) into
 * native Gemini generateContent requests and streams back events in the
 * Anthropic streaming format so the rest of the codebase is unaware.
 *
 * Uses the real Gemini REST API format (contents/parts/functionDeclarations)
 * rather than the OpenAI-compatible endpoint.
 *
 * Environment variables:
 *   CLAUDE_CODE_USE_GEMINI_NATIVE=1   — enable this provider
 *   GEMINI_API_KEY=...                — Google AI Studio API key
 *   GEMINI_MODEL=gemini-2.0-flash    — model to use
 *   GEMINI_BASE_URL=https://...       — optional custom base URL
 *                                       (default: https://generativelanguage.googleapis.com/v1beta)
 *
 * thoughtSignature: thinking models (e.g. gemini-3-flash-preview) attach a
 * thoughtSignature blob to their function-call parts. This must be echoed back
 * verbatim in the history turn when the conversation continues after a tool
 * result, otherwise the API returns 400. The shim stores it in
 * extra_content.thoughtSignature on the tool_use content block so that Claude
 * Code preserves it through its message-history layer and hands it back on the
 * next request.
 */

import type {
  AnthropicStreamEvent,
  AnthropicUsage,
  ShimCreateParams,
} from './codexShim.js'

// ---------------------------------------------------------------------------
// Gemini API types
// ---------------------------------------------------------------------------

interface GeminiPart {
  text?: string
  thoughtSignature?: string
  inlineData?: { mimeType: string; data: string }
  functionCall?: { name: string; args: Record<string, unknown> }
  functionResponse?: { name: string; response: Record<string, unknown> }
}

interface GeminiContent {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

interface GeminiFunctionDeclaration {
  name: string
  description: string
  parameters?: Record<string, unknown>
}

interface GeminiRequest {
  contents: GeminiContent[]
  systemInstruction?: { parts: [{ text: string }] }
  tools?: Array<{ functionDeclarations: GeminiFunctionDeclaration[] }>
  toolConfig?: {
    functionCallingConfig: {
      mode: 'AUTO' | 'ANY' | 'NONE'
      allowedFunctionNames?: string[]
    }
  }
  generationConfig?: {
    maxOutputTokens?: number
    temperature?: number
    topP?: number
  }
}

interface GeminiStreamChunk {
  candidates?: Array<{
    content?: { parts: GeminiPart[]; role: string }
    finishReason?: string
    index?: number
  }>
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
  error?: { code: number; message: string; status: string }
}

// ---------------------------------------------------------------------------
// Message format conversion: Anthropic → Gemini
// ---------------------------------------------------------------------------

type AnthropicMessage = {
  role: string
  message?: { role?: string; content?: unknown }
  content?: unknown
}

type AnthropicBlock = {
  type?: string
  text?: string
  thinking?: string
  id?: string
  name?: string
  input?: unknown
  tool_use_id?: string
  content?: unknown
  is_error?: boolean
  extra_content?: Record<string, unknown>
  source?: {
    type: 'base64' | 'url'
    media_type?: string
    data?: string
    url?: string
  }
}

/**
 * Build a map from tool_use id → tool name by scanning all assistant messages.
 * Gemini function responses are matched by name, not ID.
 */
function buildToolIdToNameMap(messages: AnthropicMessage[]): Map<string, string> {
  const map = new Map<string, string>()
  for (const msg of messages) {
    const inner = msg.message ?? msg
    const content = (inner as { content?: unknown }).content
    if (!Array.isArray(content)) continue
    for (const block of content as AnthropicBlock[]) {
      if (block.type === 'tool_use' && block.id && block.name) {
        map.set(block.id, block.name)
      }
    }
  }
  return map
}

function convertSystemPrompt(system: unknown): string {
  if (!system) return ''
  if (typeof system === 'string') return system
  if (Array.isArray(system)) {
    return (system as AnthropicBlock[])
      .map(b => (b.type === 'text' ? b.text ?? '' : ''))
      .join('\n\n')
  }
  return String(system)
}

/**
 * Convert a single Anthropic content block to Gemini parts.
 * The thoughtSignature stored in extra_content is echoed back on tool_use
 * so the API accepts multi-turn conversations with thinking models.
 */
function blockToParts(block: AnthropicBlock): GeminiPart[] {
  switch (block.type) {
    case 'text':
      return block.text ? [{ text: block.text }] : []
    case 'thinking':
      return block.thinking ? [{ text: `<thinking>${block.thinking}</thinking>` }] : []
    case 'image': {
      const src = block.source
      if (src?.type === 'base64' && src.media_type && src.data) {
        return [{ inlineData: { mimeType: src.media_type, data: src.data } }]
      }
      if (src?.type === 'url' && src.url) {
        return [{ text: `[Image: ${src.url}]` }]
      }
      return []
    }
    case 'tool_use': {
      const args =
        typeof block.input === 'object' && block.input !== null
          ? (block.input as Record<string, unknown>)
          : {}
      // Preserve the thoughtSignature so thinking models accept multi-turn history
      const thoughtSignature = block.extra_content?.thoughtSignature as string | undefined
      return [{
        functionCall: { name: block.name ?? 'unknown', args },
        ...(thoughtSignature ? { thoughtSignature } : {}),
      }]
    }
    default:
      return block.text ? [{ text: block.text }] : []
  }
}

/**
 * Convert a tool_result block to a Gemini functionResponse part.
 */
function toolResultToPart(
  block: AnthropicBlock,
  toolIdToName: Map<string, string>,
): GeminiPart {
  const name = toolIdToName.get(block.tool_use_id ?? '') ?? block.tool_use_id ?? 'unknown'
  const rawContent = block.content

  let resultText: string
  if (Array.isArray(rawContent)) {
    resultText = (rawContent as AnthropicBlock[])
      .map(c => c.text ?? JSON.stringify(c))
      .join('\n')
  } else if (typeof rawContent === 'string') {
    resultText = rawContent
  } else {
    resultText = JSON.stringify(rawContent ?? '')
  }

  if (block.is_error) resultText = `Error: ${resultText}`

  return {
    functionResponse: {
      name,
      response: { result: resultText },
    },
  }
}

/**
 * Convert Anthropic messages + system prompt to Gemini contents.
 *
 * Gemini rules:
 * - roles must alternate: user → model → user → model …
 * - functionResponse parts go in a user turn
 * - functionCall parts go in a model turn
 * - system prompt lives in systemInstruction (not contents)
 */
function convertMessages(
  messages: AnthropicMessage[],
  toolIdToName: Map<string, string>,
): GeminiContent[] {
  const contents: GeminiContent[] = []

  for (const msg of messages) {
    const inner = msg.message ?? msg
    const role = ((inner as { role?: string }).role ?? msg.role) as string
    const content = (inner as { content?: unknown }).content

    if (role === 'assistant') {
      if (!Array.isArray(content)) {
        const text = typeof content === 'string' ? content : JSON.stringify(content ?? '')
        if (text) contents.push({ role: 'model', parts: [{ text }] })
        continue
      }

      const parts: GeminiPart[] = []
      for (const block of content as AnthropicBlock[]) {
        if (block.type === 'tool_result') continue
        parts.push(...blockToParts(block))
      }
      if (parts.length > 0) contents.push({ role: 'model', parts })
    } else if (role === 'user') {
      if (!Array.isArray(content)) {
        const text = typeof content === 'string' ? content : JSON.stringify(content ?? '')
        if (text) contents.push({ role: 'user', parts: [{ text }] })
        continue
      }

      const toolResults = (content as AnthropicBlock[]).filter(b => b.type === 'tool_result')
      const other = (content as AnthropicBlock[]).filter(b => b.type !== 'tool_result')

      const userParts: GeminiPart[] = []

      // functionResponse parts
      if (toolResults.length > 0) {
        userParts.push(...toolResults.map(tr => toolResultToPart(tr, toolIdToName)))
      }

      // Remaining user content
      for (const block of other) userParts.push(...blockToParts(block))

      if (userParts.length > 0) {
        contents.push({ role: 'user', parts: userParts })
      }
    }
  }

  return contents
}

/**
 * Fields that the Gemini function-declaration parameters schema does not accept.
 * Sending them causes a 400 "Unknown name" error.
 */
const GEMINI_UNSUPPORTED_SCHEMA_KEYS = new Set([
  '$schema',
  'additionalProperties',
  'exclusiveMinimum',
  'exclusiveMaximum',
  'contentMediaType',
  'contentEncoding',
  'unevaluatedProperties',
])

/**
 * Recursively strip schema keys that Gemini rejects from JSON Schema objects.
 */
function sanitizeSchemaForGemini(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(sanitizeSchemaForGemini)
  }
  if (schema !== null && typeof schema === 'object') {
    const out: Record<string, unknown> = {}
    for (const [key, val] of Object.entries(schema as Record<string, unknown>)) {
      if (GEMINI_UNSUPPORTED_SCHEMA_KEYS.has(key)) continue
      out[key] = sanitizeSchemaForGemini(val)
    }
    return out
  }
  return schema
}

function convertTools(
  tools: Array<{ name: string; description?: string; input_schema?: Record<string, unknown> }>,
): Array<{ functionDeclarations: GeminiFunctionDeclaration[] }> {
  const decls: GeminiFunctionDeclaration[] = tools
    .filter(t => t.name !== 'ToolSearchTool')
    .map(t => ({
      name: t.name,
      description: t.description ?? '',
      ...(t.input_schema
        ? { parameters: sanitizeSchemaForGemini(t.input_schema) as Record<string, unknown> }
        : {}),
    }))

  return decls.length > 0 ? [{ functionDeclarations: decls }] : []
}

function convertToolChoice(toolChoice: unknown): GeminiRequest['toolConfig'] | undefined {
  if (!toolChoice) return undefined
  const tc = toolChoice as { type?: string; name?: string }
  let mode: 'AUTO' | 'ANY' | 'NONE' = 'AUTO'
  const allowedFunctionNames: string[] = []

  if (tc.type === 'none') mode = 'NONE'
  else if (tc.type === 'any') mode = 'ANY'
  else if (tc.type === 'tool' && tc.name) { mode = 'ANY'; allowedFunctionNames.push(tc.name) }

  return {
    functionCallingConfig: {
      mode,
      ...(allowedFunctionNames.length > 0 ? { allowedFunctionNames } : {}),
    },
  }
}

// ---------------------------------------------------------------------------
// Build the Gemini request body
// ---------------------------------------------------------------------------

function buildGeminiRequest(params: ShimCreateParams): GeminiRequest {
  const messages = params.messages as AnthropicMessage[]
  const toolIdToName = buildToolIdToNameMap(messages)
  const contents = convertMessages(messages, toolIdToName)
  const sysText = convertSystemPrompt(params.system)

  const req: GeminiRequest = { contents }

  if (sysText) req.systemInstruction = { parts: [{ text: sysText }] }

  if (params.tools && (params.tools as unknown[]).length > 0) {
    const toolDefs = convertTools(
      params.tools as Array<{
        name: string
        description?: string
        input_schema?: Record<string, unknown>
      }>,
    )
    if (toolDefs.length > 0) {
      req.tools = toolDefs
      const toolConfig = convertToolChoice(params.tool_choice)
      if (toolConfig) req.toolConfig = toolConfig
    }
  }

  const genConfig: GeminiRequest['generationConfig'] = {}
  if (params.max_tokens !== undefined) genConfig.maxOutputTokens = params.max_tokens
  if (params.temperature !== undefined) genConfig.temperature = params.temperature
  if (params.top_p !== undefined) genConfig.topP = params.top_p
  if (Object.keys(genConfig).length > 0) req.generationConfig = genConfig

  return req
}

// ---------------------------------------------------------------------------
// Collect all SSE chunks from the response body
// ---------------------------------------------------------------------------

async function collectSSEChunks(response: Response): Promise<GeminiStreamChunk[]> {
  const chunks: GeminiStreamChunk[] = []
  const reader = response.body?.getReader()
  if (!reader) return chunks

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed === 'data: [DONE]') continue
      if (!trimmed.startsWith('data: ')) continue

      let chunk: GeminiStreamChunk
      try {
        chunk = JSON.parse(trimmed.slice(6))
      } catch {
        continue
      }

      if (chunk.error) {
        throw new Error(`Gemini API error ${chunk.error.code}: ${chunk.error.message}`)
      }
      chunks.push(chunk)
    }
  }

  return chunks
}

// ---------------------------------------------------------------------------
// Streaming: Gemini SSE → Anthropic stream events
//
// Thinking models (e.g. gemini-3-flash-preview) split their response across
// two chunks:
//   chunk 1 — functionCall (or text) parts, no finishReason
//   chunk 2 — empty text part carrying thoughtSignature, finishReason=STOP
//
// We collect ALL chunks before emitting any events so we can:
//  1. Associate thoughtSignatures with their functionCall parts
//  2. Determine the correct stop_reason (tool_use vs end_turn) even when
//     finishReason and functionCalls arrive in different chunks
// ---------------------------------------------------------------------------

interface PendingPart {
  kind: 'text' | 'functionCall'
  text?: string
  name?: string
  args?: Record<string, unknown>
  thoughtSignature?: string
}

function mapFinishReason(
  finishReason: string | undefined,
  hasFunctionCalls: boolean,
): 'end_turn' | 'tool_use' | 'max_tokens' {
  if (hasFunctionCalls) return 'tool_use'
  if (finishReason === 'MAX_TOKENS') return 'max_tokens'
  return 'end_turn'
}

function makeMessageId(): string {
  return `msg_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`
}

async function* geminiStreamToAnthropic(
  response: Response,
  model: string,
): AsyncGenerator<AnthropicStreamEvent> {
  const messageId = makeMessageId()

  // Collect all SSE chunks so we can associate thoughtSignatures with functionCalls
  const rawChunks = await collectSSEChunks(response)

  const pendingParts: PendingPart[] = []
  const pendingSignatures: string[] = []
  let finishReason: string | undefined
  let finalUsage: Partial<AnthropicUsage> | undefined

  for (const chunk of rawChunks) {
    if (chunk.usageMetadata) {
      finalUsage = {
        input_tokens: chunk.usageMetadata.promptTokenCount ?? 0,
        output_tokens: chunk.usageMetadata.candidatesTokenCount ?? 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      }
    }

    const candidate = chunk.candidates?.[0]
    if (!candidate) continue
    if (candidate.finishReason) finishReason = candidate.finishReason

    for (const part of candidate.content?.parts ?? []) {
      if (part.thoughtSignature) {
        // Collect signatures; they'll be attached to functionCall parts below
        pendingSignatures.push(part.thoughtSignature)
      }
      if (part.functionCall) {
        pendingParts.push({
          kind: 'functionCall',
          name: part.functionCall.name,
          args: part.functionCall.args ?? {},
        })
      } else if (part.text) {
        // Skip empty text parts (they only carry thoughtSignature, handled above)
        pendingParts.push({ kind: 'text', text: part.text })
      }
    }
  }

  // Attach collected thoughtSignatures to functionCall parts in order
  let sigIdx = 0
  for (const part of pendingParts) {
    if (part.kind === 'functionCall' && sigIdx < pendingSignatures.length) {
      part.thoughtSignature = pendingSignatures[sigIdx++]
    }
  }

  const hasFunctionCalls = pendingParts.some(p => p.kind === 'functionCall')
  const stopReason = mapFinishReason(finishReason, hasFunctionCalls)

  // Emit message_start
  yield {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  }

  let contentBlockIndex = 0
  for (const part of pendingParts) {
    if (part.kind === 'text') {
      yield {
        type: 'content_block_start',
        index: contentBlockIndex,
        content_block: { type: 'text', text: '' },
      }
      yield {
        type: 'content_block_delta',
        index: contentBlockIndex,
        delta: { type: 'text_delta', text: part.text! },
      }
      yield { type: 'content_block_stop', index: contentBlockIndex }
      contentBlockIndex++
    } else if (part.kind === 'functionCall') {
      const toolId = `call_${Math.random().toString(36).slice(2)}`
      yield {
        type: 'content_block_start',
        index: contentBlockIndex,
        content_block: {
          type: 'tool_use',
          id: toolId,
          name: part.name!,
          input: {},
          // Store thoughtSignature so it survives Claude Code's message-history
          // layer and is echoed back when building the next Gemini request
          ...(part.thoughtSignature
            ? { extra_content: { thoughtSignature: part.thoughtSignature } }
            : {}),
        },
      }
      yield {
        type: 'content_block_delta',
        index: contentBlockIndex,
        delta: {
          type: 'input_json_delta',
          partial_json: JSON.stringify(part.args!),
        },
      }
      yield { type: 'content_block_stop', index: contentBlockIndex }
      contentBlockIndex++
    }
  }

  yield {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    ...(finalUsage ? { usage: finalUsage } : {}),
  }

  yield { type: 'message_stop' }
}

// ---------------------------------------------------------------------------
// Non-streaming response conversion
// ---------------------------------------------------------------------------

function convertNonStreamingResponse(
  data: {
    candidates?: Array<{
      content?: { parts: GeminiPart[]; role: string }
      finishReason?: string
    }>
    usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number }
    error?: { code: number; message: string }
  },
  model: string,
) {
  if (data.error) {
    throw new Error(`Gemini API error ${data.error.code}: ${data.error.message}`)
  }

  const candidate = data.candidates?.[0]
  const parts = candidate?.content?.parts ?? []
  const content: Array<Record<string, unknown>> = []

  // Collect thoughtSignatures to attach to functionCalls
  const signatures = parts.filter(p => p.thoughtSignature).map(p => p.thoughtSignature!)
  let sigIdx = 0

  for (const part of parts) {
    if (part.text) {
      content.push({ type: 'text', text: part.text })
    } else if (part.functionCall) {
      const thoughtSignature = signatures[sigIdx++]
      content.push({
        type: 'tool_use',
        id: `call_${Math.random().toString(36).slice(2)}`,
        name: part.functionCall.name,
        input: part.functionCall.args ?? {},
        ...(thoughtSignature ? { extra_content: { thoughtSignature } } : {}),
      })
    }
  }

  const hasFunctionCalls = parts.some(p => p.functionCall !== undefined)
  const stopReason = mapFinishReason(candidate?.finishReason, hasFunctionCalls)

  return {
    id: makeMessageId(),
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: data.usageMetadata?.promptTokenCount ?? 0,
      output_tokens: data.usageMetadata?.candidatesTokenCount ?? 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
  }
}

// ---------------------------------------------------------------------------
// HTTP request
// ---------------------------------------------------------------------------

function getGeminiBaseUrl(): string {
  // Strip /openai suffix if the user copied their GEMINI_BASE_URL from the
  // OpenAI-compat path — we're talking to the native endpoint here
  return (
    process.env.GEMINI_BASE_URL?.replace(/\/openai\/?$/, '') ||
    'https://generativelanguage.googleapis.com/v1beta'
  )
}

async function doGeminiRequest(
  params: ShimCreateParams,
  defaultHeaders: Record<string, string>,
  signal?: AbortSignal,
): Promise<Response> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || ''
  const model = params.model
  const baseUrl = getGeminiBaseUrl()
  const action = params.stream ? 'streamGenerateContent' : 'generateContent'
  const altParam = params.stream ? '?alt=sse' : ''
  const url = `${baseUrl}/models/${model}:${action}${altParam}`

  const body = buildGeminiRequest(params)

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...defaultHeaders,
  }
  if (apiKey) headers['x-goog-api-key'] = apiKey

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    const errorBody = await response.text().catch(() => 'unknown error')
    throw new Error(`Gemini API error ${response.status}: ${errorBody}`)
  }

  return response
}

// ---------------------------------------------------------------------------
// Shim stream wrapper
// ---------------------------------------------------------------------------

class GeminiShimStream {
  private generator: AsyncGenerator<AnthropicStreamEvent>
  controller = new AbortController()

  constructor(generator: AsyncGenerator<AnthropicStreamEvent>) {
    this.generator = generator
  }

  async *[Symbol.asyncIterator]() {
    yield* this.generator
  }
}

// ---------------------------------------------------------------------------
// Shim client — duck-types as Anthropic SDK
// ---------------------------------------------------------------------------

class GeminiShimMessages {
  private defaultHeaders: Record<string, string>

  constructor(defaultHeaders: Record<string, string>) {
    this.defaultHeaders = defaultHeaders
  }

  create(
    params: ShimCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ) {
    const headers = { ...this.defaultHeaders, ...(options?.headers ?? {}) }

    const promise = (async () => {
      const response = await doGeminiRequest(params, headers, options?.signal)

      if (params.stream) {
        return new GeminiShimStream(geminiStreamToAnthropic(response, params.model))
      }

      const data = await response.json()
      return convertNonStreamingResponse(data, params.model)
    })()

    ;(promise as unknown as Record<string, unknown>).withResponse = async () => {
      const data = await promise
      return { data, response: new Response(), request_id: makeMessageId() }
    }

    return promise
  }
}

class GeminiShimBeta {
  messages: GeminiShimMessages

  constructor(defaultHeaders: Record<string, string>) {
    this.messages = new GeminiShimMessages(defaultHeaders)
  }
}

export function createGeminiNativeShimClient(options: {
  defaultHeaders?: Record<string, string>
  maxRetries?: number
  timeout?: number
}): unknown {
  const beta = new GeminiShimBeta(options.defaultHeaders ?? {})
  return { beta, messages: beta.messages }
}
