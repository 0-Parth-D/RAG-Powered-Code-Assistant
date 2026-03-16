#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::string> tokenize(std::string s) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while(iss >> token) {
        std::transform(token.begin(), token.end(), token.begin(),
                [](unsigned char c){ return std::tolower(c); });
        if (token.empty()) continue;
        result.push_back(token);
    }
    return result;
}

/* pybind11 bindings */
PYBIND11_MODULE(fast_tokenizer, m) {
    m.doc() = "Fast C++ tokenizer plugin for RAG Code Assistant"; 
    m.def("tokenize", &tokenize, "A function that splits a string by whitespace and lowercases it");
}