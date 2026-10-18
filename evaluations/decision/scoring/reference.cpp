// Pinned llama.cpp CPU client. Each line: label_count label_ids... prompt_ids...
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp); if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=512;
    cp.n_threads=4; cp.n_threads_batch=4;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx = llama_init_from_model(model, cp); if (!ctx) return 3;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ifstream input(argv[2]); std::ofstream out(argv[3]);
    if (!input || !out) return 4;
    out << std::setprecision(17);
    std::string line; int row=0;
    while (std::getline(input,line)) {
        std::istringstream in(line); int n=0, t;
        if (!(in >> n) || n<2 || n>8) return 5;
        std::vector<llama_token> labels(n), ids;
        for (auto& label: labels) if (!(in >> label) || label<0 || label>=vocab) return 6;
        auto unique=labels; std::sort(unique.begin(),unique.end());
        if (std::adjacent_find(unique.begin(),unique.end())!=unique.end()) return 7;
        while (in >> t) { if(t<0 || t>=vocab) return 8; ids.push_back(t); }
        if (!in.eof() || ids.empty() || ids.size()>512) return 9;
        llama_memory_clear(llama_get_memory(ctx),true);
        auto start=std::chrono::steady_clock::now();
        auto batch=llama_batch_get_one(ids.data(),static_cast<int32_t>(ids.size()));
        if (llama_decode(ctx,batch)) { out << "{\"error\":\"decode failed\"}\n"; continue; }
        auto* logits=llama_get_logits_ith(ctx,-1);
        if (!logits || !std::all_of(logits,logits+vocab,[](float x){return std::isfinite(x);})) {
            out << "{\"error\":\"nonfinite logits\"}\n"; continue;
        }
        const double max=*std::max_element(logits,logits+vocab);
        double total=0, label_total=0;
        for (int i=0;i<vocab;i++) total+=std::exp(static_cast<double>(logits[i])-max);
        for (auto label:labels) label_total+=std::exp(static_cast<double>(logits[label])-max);
        std::vector<int> top(vocab); std::iota(top.begin(),top.end(),0);
        std::partial_sort(top.begin(),top.begin()+5,top.end(),[&](int a,int b){
            return logits[a]==logits[b] ? a<b : logits[a]>logits[b];
        });
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        out << "{\"elapsedMs\":" << ms << ",\"labelMass\":" << label_total/total << ",\"logits\":[";
        for(int i=0;i<n;i++) { if(i) out << ','; out << logits[labels[i]]; }
        out << "],\"top\":[";
        for(int i=0;i<5;i++) { if(i) out << ','; out << "{\"tokenId\":" << top[i]
            << ",\"probability\":" << std::exp(static_cast<double>(logits[top[i]])-max)/total << '}'; }
        out << "]}\n"; out.flush();
        std::cerr << "attempt " << ++row << ": " << ms << " ms\n";
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
}
