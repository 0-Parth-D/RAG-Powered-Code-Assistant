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

// Inside your C++ code
size_t count_tokens(const std::string& text) {
    // Assuming tokenize() is your existing function that returns std::vector<std::string>
    std::vector<std::string> tokens = tokenize(text);
    return tokens.size();
}

/* pybind11 bindings */
PYBIND11_MODULE(fast_tokenizer, m) {
    m.doc() = "Fast C++ tokenizer plugin for RAG Code Assistant"; 
    m.def("tokenize", &tokenize, "A function that splits a string by whitespace and lowercases it");
    m.def("count_tokens", &count_tokens, "Returns the number of tokens in the text");
}