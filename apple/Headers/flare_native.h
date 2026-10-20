#ifndef FLARE_NATIVE_H
#define FLARE_NATIVE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All calls on an engine must be serialized, including close. Every returned
 * string is owned by Rust and must be freed with flare_native_string_free.
 * JSON responses contain either {"value":...} or {"error":"..."}. */
typedef struct NativeEngine NativeEngine;
NativeEngine *flare_native_open(const char *model, const char *tokenizer,
                               uint32_t context, char **status);
char *flare_native_begin(NativeEngine *engine, const char *request_json);
char *flare_native_prefill(NativeEngine *engine);
char *flare_native_next(NativeEngine *engine);
char *flare_native_reset(NativeEngine *engine);
void flare_native_close(NativeEngine *engine);
void flare_native_string_free(char *text);
#ifdef __cplusplus
}
#endif
#endif
