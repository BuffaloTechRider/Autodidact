import OpenAI from 'openai';
import { safeValidateChatResponse } from '../schemas.js';
import type {
    AutodidactError,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ILLMClient,
} from '../types.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export interface LLMClientConfig {
    baseUrl: string;
    apiKey: string;
    model: string;
    timeoutMs?: number;
}

function makeError(
    code: string,
    message: string,
    details?: unknown,
): AutodidactError {
    return {
        code,
        message,
        component: 'LLMClient',
        details,
        timestamp: nowISO(),
    };
}

export class LLMClient implements ILLMClient {
    private readonly client: OpenAI;
    private readonly model: string;
    private readonly timeoutMs: number;
    private readonly logger: Logger;

    constructor(config: LLMClientConfig, logger: Logger = defaultLogger) {
        this.client = new OpenAI({
            baseURL: config.baseUrl,
            apiKey: config.apiKey,
            timeout: config.timeoutMs ?? 30_000,
        });
        this.model = config.model;
        this.timeoutMs = config.timeoutMs ?? 30_000;
        this.logger = logger;
    }

    async chat(
        messages: ChatMessage[],
        options?: ChatOptions,
    ): Promise<ChatResponse> {
        try {
            this.logger.debug('LLMClient.chat: sending request', {
                model: options?.model ?? this.model,
                messageCount: messages.length,
            });

            const response = await this.client.chat.completions.create({
                model: options?.model ?? this.model,
                messages: messages.map((m) => ({
                    role: m.role,
                    content: m.content,
                })),
                temperature: options?.temperature,
                max_tokens: options?.maxTokens,
            });

            const choice = response.choices?.[0];
            const raw = {
                content: choice?.message?.content ?? '',
                usage: {
                    promptTokens: response.usage?.prompt_tokens ?? 0,
                    completionTokens: response.usage?.completion_tokens ?? 0,
                },
                model: response.model ?? options?.model ?? this.model,
            };

            const result = safeValidateChatResponse(raw);
            if (!result.success) {
                this.logger.error(
                    'LLMClient.chat: malformed response',
                    result.error,
                );
                throw makeError(
                    'LLM_MALFORMED_RESPONSE',
                    'LLM response failed Zod validation',
                    { raw, zodError: result.error.format() },
                );
            }

            this.logger.debug('LLMClient.chat: success', {
                model: result.data.model,
            });
            return result.data;
        } catch (err) {
            if (isAutodidactError(err)) {
                throw err;
            }
            if (isTimeoutError(err)) {
                this.logger.error('LLMClient.chat: timeout', {
                    timeoutMs: this.timeoutMs,
                });
                throw makeError('LLM_TIMEOUT', `Request timed out after ${this.timeoutMs}ms`, {
                    timeoutMs: this.timeoutMs,
                });
            }
            this.logger.error('LLMClient.chat: unexpected error', err);
            throw makeError(
                'LLM_REQUEST_FAILED',
                err instanceof Error ? err.message : 'Unknown error',
                { originalError: String(err) },
            );
        }
    }

    async embed(text: string): Promise<number[]> {
        try {
            this.logger.debug('LLMClient.embed: sending request');

            const response = await this.client.embeddings.create({
                model: this.model,
                input: text,
            });

            const embedding = response.data?.[0]?.embedding;
            if (!Array.isArray(embedding)) {
                this.logger.error('LLMClient.embed: malformed response');
                throw makeError(
                    'LLM_MALFORMED_RESPONSE',
                    'Embeddings response missing data[0].embedding array',
                    { raw: response },
                );
            }

            this.logger.debug('LLMClient.embed: success', {
                dimensions: embedding.length,
            });
            return embedding;
        } catch (err) {
            if (isAutodidactError(err)) {
                throw err;
            }
            if (isTimeoutError(err)) {
                this.logger.error('LLMClient.embed: timeout', {
                    timeoutMs: this.timeoutMs,
                });
                throw makeError('LLM_TIMEOUT', `Embed request timed out after ${this.timeoutMs}ms`, {
                    timeoutMs: this.timeoutMs,
                });
            }
            this.logger.error('LLMClient.embed: unexpected error', err);
            throw makeError(
                'LLM_REQUEST_FAILED',
                err instanceof Error ? err.message : 'Unknown error',
                { originalError: String(err) },
            );
        }
    }
}

function isAutodidactError(err: unknown): err is AutodidactError {
    return (
        typeof err === 'object' &&
        err !== null &&
        'code' in err &&
        'component' in err &&
        'timestamp' in err
    );
}

function isTimeoutError(err: unknown): boolean {
    if (err instanceof Error) {
        const msg = err.message.toLowerCase();
        if (msg.includes('timeout') || msg.includes('timed out')) {
            return true;
        }
        // OpenAI SDK uses APIConnectionTimeoutError
        if (err.constructor.name === 'APIConnectionTimeoutError') {
            return true;
        }
    }
    return false;
}
