#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "idownloader.hpp"

using std::string;
using std::tuple;
using std::unordered_set;
using std::vector;

namespace jmnel::iex {

    void get_top_download_list();
    //    void download( string const url );

    class downloader final : public idownloader {
    private:
        bool m_is_stopping = false;

        void download_dump( string url );

        struct endpoint_item {
            const string date;
            const string link;
            const string protocol;
            const size_t size;
            const string version;
        };

    public:
        downloader();
        virtual ~downloader() final;
        virtual void request_stop() final;
        virtual void do_download() final;

        friend std::ostream &operator<<( std::ostream &, endpoint_item const & );

    private:
        vector<tuple<string, string>> get_urls();
    };

    inline std::ostream &operator<<( std::ostream &os, downloader::endpoint_item const &it ) {
        os << it.date << ", ";
        os << it.version;
        return os;
    }

}  // namespace jmnel::iex
