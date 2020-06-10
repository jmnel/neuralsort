#pragma once

#include <filesystem>
#include <string>

using std::string;
namespace fs = std::filesystem;

namespace jmnel::iex {

    const string hist_endpoint = "https://iextrading.com/api/1.0/hist";
    const fs::path data_directory = "/home/jacques/repos/jmnel/neuralsort/realtime/data";
    const auto database_path = data_directory / "iex.sqlite3";

}  // namespace jmnel::iex
