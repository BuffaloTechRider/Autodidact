/**
 * Return the current time as an ISO 8601 string.
 */
export function nowISO(): string {
    return new Date().toISOString();
}

/**
 * Compute the number of hours elapsed since the given ISO timestamp.
 */
export function hoursSince(isoTimestamp: string): number {
    const then = new Date(isoTimestamp).getTime();
    const now = Date.now();
    return (now - then) / (1000 * 60 * 60);
}

/**
 * Compute the number of milliseconds elapsed since the given ISO timestamp.
 */
export function msSince(isoTimestamp: string): number {
    const then = new Date(isoTimestamp).getTime();
    return Date.now() - then;
}
