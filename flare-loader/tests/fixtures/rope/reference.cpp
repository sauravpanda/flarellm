#include "llama.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp); if (!model) return 2;
    auto cp = llama_context_default_params(); cp.n_ctx=256; cp.n_batch=256; cp.n_ubatch=256; cp.n_threads=4; cp.n_threads_batch=4; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx = llama_init_from_model(model, cp); if (!ctx) return 3;
    std::ifstream input(argv[2]); std::vector<llama_token> ids; int t; while(input>>t) ids.push_back(t);
    auto batch=llama_batch_get_one(ids.data(), ids.size()); if(llama_decode(ctx,batch)) return 4;
    int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream out(argv[3]); out<<std::setprecision(9)<<"{\"steps\":[";
    for(int step=0;step<16;step++) {
        auto* logits=llama_get_logits_ith(ctx,-1); int token=std::max_element(logits,logits+vocab)-logits;
        if(step) out<<","; out<<"{\"token\":"<<token<<",\"logits\":[";
        for(int i=0;i<vocab;i++) { if(i) out<<","; out<<logits[i]; } out<<"]}";
        if(token==2) break;
        llama_token next=token; auto b=llama_batch_get_one(&next,1); if(llama_decode(ctx,b)) return 5;
    }
    out<<"]}"; llama_free(ctx); llama_model_free(model); llama_backend_free();
}
