// Build against the pinned llama.cpp CPU library; one line of prompt IDs per case.
#include "llama.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp); if (!model) return 2;
    auto cp = llama_context_default_params(); cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=512;
    cp.n_threads=4; cp.n_threads_batch=4; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx = llama_init_from_model(model, cp); if (!ctx) return 3;
    std::ifstream input(argv[2]); std::ofstream out(argv[3]); std::string line;
    out << std::setprecision(9) << "["; bool first=true;
    while (std::getline(input,line)) {
        std::istringstream in(line); std::vector<llama_token> ids; int t;
        while(in >> t) ids.push_back(t);
        if(ids.empty() || ids.size()>512) return 4;
        llama_memory_clear(llama_get_memory(ctx),true);
        auto batch=llama_batch_get_one(ids.data(),ids.size()); if(llama_decode(ctx,batch)) return 5;
        auto* logits=llama_get_logits_ith(ctx,-1);
        if (!first) out << ","; first=false; out << "[";
        for(int i=32;i<40;i++) { if(i>32) out << ","; out << logits[i]; }
        out << "]";
    }
    out << "]"; llama_free(ctx); llama_model_free(model); llama_backend_free();
}
