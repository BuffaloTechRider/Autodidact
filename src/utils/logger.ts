/**
 * Simple pluggable logger interface. Defaults to console.
 */
export interface Logger {
    info(message: string, ...args: unknown[]): void;
    warn(message: string, ...args: unknown[]): void;
    error(message: string, ...args: unknown[]): void;
    debug(message: string, ...args: unknown[]): void;
}

/**
 * Default logger implementation using console.
 */
export const defaultLogger: Logger = {
    info: (message, ...args) => console.log(`[INFO] ${message}`, ...args),
    warn: (message, ...args) => console.warn(`[WARN] ${message}`, ...args),
    error: (message, ...args) => console.error(`[ERROR] ${message}`, ...args),
    debug: (message, ...args) => console.debug(`[DEBUG] ${message}`, ...args),
};

/**
 * Create a silent logger that discards all output.
 */
export function createSilentLogger(): Logger {
    const noop = () => { };
    return { info: noop, warn: noop, error: noop, debug: noop };
}
