#include "iex_downloader.hpp"

#include <curl/curl.h>
#include <curl/easy.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <iostream>
#include <new>
#include <sstream>

#include "common.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::chrono::year_month_day;

namespace pt = boost::property_tree;

namespace jmnel::iex {

    size_t curl_write( void* contents, size_t size, size_t nmemb, std::string* s ) {
        const size_t new_length = size * nmemb;
        try {
            s->append( (char*)contents, new_length );
        } catch( std::bad_alloc& e ) {
            return 0;
        }
        return new_length;
    }

    void get_trading_days() {
        auto curl = curl_easy_init();
        if( !curl ) {
            cerr << "Failed to initialize curl." << endl;
            std::terminate();
        }

        curl_easy_setopt( curl,
                          CURLOPT_URL,
                          arc::jmnel::hist_endpoint.c_str() );
        //        auto devnull = fopen( "/dev/null", "w+" );
        curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, curl_write );
        string response = "";
        curl_easy_setopt( curl, CURLOPT_WRITEDATA, &response );

        auto result = curl_easy_perform( curl );

        std::stringstream ss;
        ss << response;
        curl_easy_cleanup( curl );

        pt::ptree root;
        pt::read_json( ss, root );

        for( auto a : root ) {
            cout << a.first << ": " << endl;
        }
    }

}  // namespace jmnel::iex
