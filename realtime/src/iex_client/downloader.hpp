#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "idownloader.hpp"

using std::string;
using std::tuple;
using std::vector;

namespace jmnel::iex {

    void get_top_download_list();
    //    void download( string const url );

    class downloader final : public idownloader {
    private:
        bool m_is_stopping = false;

    public:
        downloader();
        virtual ~downloader() final;
        virtual void request_stop() final;
        virtual void start_download() final;

    private:
        vector<tuple<string, string>> get_urls();
    };

}  // namespace jmnel::iex
