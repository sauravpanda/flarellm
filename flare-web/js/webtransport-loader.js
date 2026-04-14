/**
 * Progressive model loader using WebTransport for parallel stream downloads.
 *
 * WebTransport is built on HTTP/3 QUIC and allows multiple bidirectional
 * streams over a single connection, avoiding the head-of-line blocking that
 * plagues HTTP/1.1/2 range requests when downloading large model files.
 *
 * NOTE ON SERVER SUPPORT:
 *   For parallel stream downloads to actually happen, the server must:
 *     1. Speak HTTP/3 and expose a WebTransport endpoint at the model URL.
 *     2. Accept a per-stream protocol where the client sends a byte range
 *        (e.g. as a small framed message) and the server streams those bytes
 *        back on the same bidirectional stream.
 *   No such server is shipped with Flare today. Until one exists, this loader
 *   transparently falls back to `fetch()` with streaming, which still gives
 *   progressive load + progress callbacks over HTTP/1.1 or HTTP/2.
 *
 * Usage:
 *   const loader = new WebTransportLoader('https://example.com/model.gguf', 4);
 *   const bytes = await loader.load((loaded, total) => {
 *     console.log(`${loaded} / ${total}`);
 *   });
 */
export class WebTransportLoader {
    /**
     * @param {string} url   Absolute URL to the model file.
     * @param {number} numStreams  Number of parallel streams to attempt when
     *                             WebTransport + a cooperating server are
     *                             available. Ignored by the fetch fallback.
     */
    constructor(url, numStreams = 4) {
        this.url = url;
        this.numStreams = numStreams;
    }

    /**
     * Load the model bytes, invoking `onProgress(loaded, total)` as data
     * arrives. Returns a Uint8Array containing the full file.
     *
     * @param {(loaded: number, total: number) => void} [onProgress]
     * @returns {Promise<Uint8Array>}
     */
    async load(onProgress) {
        if (typeof WebTransport === 'undefined') {
            // Browser has no WebTransport at all — use fetch streaming.
            return this.loadViaFetch(onProgress);
        }

        try {
            const wt = new WebTransport(this.url);
            await wt.ready;

            // Server-side protocol for parallel range streaming is not yet
            // standardized in this project. Once a Flare WebTransport server
            // exists, this block should:
            //   1. HEAD the resource (or open a control stream) to learn the
            //      total byte length.
            //   2. Open `this.numStreams` bidirectional streams via
            //      `wt.createBidirectionalStream()`.
            //   3. Send a framed { offset, length } request on each stream.
            //   4. Reassemble the chunks in offset order as they arrive,
            //      invoking `onProgress` on every chunk.
            //
            // Until that server exists, we close the WebTransport session and
            // fall back to fetch — the browser has already done a QUIC
            // handshake, which is wasted work but harmless. We log so that
            // anyone instrumenting this path can see it.
            console.info(
                'WebTransport session opened, but no parallel-range server ' +
                'protocol is implemented yet; falling back to fetch().'
            );
            await wt.close();
        } catch (e) {
            console.warn('WebTransport failed, falling back to fetch:', e);
        }

        return this.loadViaFetch(onProgress);
    }

    /**
     * Fallback path: stream the body via `fetch()` and report progress from
     * the Content-Length header. Works on any HTTP/1.1+ origin.
     *
     * @param {(loaded: number, total: number) => void} [onProgress]
     * @returns {Promise<Uint8Array>}
     */
    async loadViaFetch(onProgress) {
        const response = await fetch(this.url);
        if (!response.ok) {
            throw new Error(
                `Failed to fetch model: ${response.status} ${response.statusText}`
            );
        }
        if (!response.body) {
            // Non-streaming environment (very old browser or CORS-restricted).
            const buf = new Uint8Array(await response.arrayBuffer());
            if (onProgress) onProgress(buf.length, buf.length);
            return buf;
        }

        const contentLength = parseInt(
            response.headers.get('content-length') || '0',
            10
        );
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            if (onProgress) onProgress(loaded, contentLength);
        }

        // Concatenate chunks into a single contiguous Uint8Array.
        const result = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }
}
