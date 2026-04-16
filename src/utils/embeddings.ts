/**
 * Serialize a number[] embedding to a Buffer (Float32Array) for SQLite BLOB storage.
 */
export function serializeEmbedding(embedding: number[]): Buffer {
    const float32 = new Float32Array(embedding);
    return Buffer.from(float32.buffer);
}

/**
 * Deserialize a SQLite BLOB (Buffer) back to a number[] embedding.
 */
export function deserializeEmbedding(blob: Buffer): number[] {
    const float32 = new Float32Array(
        blob.buffer,
        blob.byteOffset,
        blob.byteLength / Float32Array.BYTES_PER_ELEMENT
    );
    return Array.from(float32);
}
