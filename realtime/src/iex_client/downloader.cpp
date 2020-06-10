#include <boost/iostreams/categories.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <new>
#include <sstream>
#include <tuple>

#include <curl/curl.h>
#include <curl/easy.h>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <nlohmann/json.hpp>

#include "common.hpp"
#include "decoder.hpp"
#include "downloader.hpp"
#include "message.hpp"

namespace fs = std::filesystem;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios_base;
using std::ofstream;
using std::tuple;

using boost::iostreams::filtering_streambuf;

using json = nlohmann::json;

namespace jmnel::iex {

    // -- Constructor --
    downloader::downloader() {
    }

    // -- Destructor --
    downloader::~downloader() {
        cout << "downloader destroyed" << endl;
    }

    // -- request_stop function --
    void downloader::request_stop() {
        cout << "graceful stop requested." << endl;
        m_is_stopping = true;
    }

    // -- start_download function --
    void downloader::start_download() {
        const auto urls = get_urls();

        for( size_t idx = 0; idx < urls.size(); ++idx ) {
            if( m_is_stopping ) {
                cout << "exit signal received. stopping." << endl;
                break;
            }
            cout << "downloading " << idx + 1 << " of ";
            cout << urls.size() << ": " << std::get<0>( urls[idx] );
            cout << " ...\n";

            auto curl = curl_easy_init();
            if( !curl ) {
                cerr << "  failed to initialize curl.\n";
                std::terminate();
            }

            const auto endpoint = std::get<1>( urls[idx] );
            const auto filename_gz = std::get<0>( urls[idx] ) + ".pcap.gz";
            const auto filename = std::get<0>( urls[idx] ) + ".pcap";

            const auto out_path_gz = data_directory / "raw" / filename_gz;
            const auto out_path = data_directory / "raw" / filename;

            if( fs::exists( out_path_gz ) or fs::exists( out_path ) ) {
                cout << "  " << filename_gz << " already exists.\n";
                continue;
            }

            FILE* out_c_file = nullptr;
            if( out_c_file = fopen( out_path_gz.c_str(), "w" ); !out_c_file ) {
                cerr << "  failed to create file for download." << endl;
                std::abort();
            }

            curl_easy_setopt( curl,
                              CURLOPT_URL,
                              std::get<1>( urls[idx] ).c_str() );
            string response = "";
            curl_easy_setopt( curl, CURLOPT_WRITEDATA, out_c_file );
            curl_easy_perform( curl );
            curl_easy_cleanup( curl );
            curl = nullptr;
            fclose( out_c_file );
            cout << "  download complete.\n\n";

            // Uncompress downloaded gzip file.
            cout << "  uncompressing...\n";
            std::ifstream gzip_file( out_path_gz, ios_base::in | ios_base::binary );
            filtering_streambuf<boost::iostreams::input> in_stream;
            std::ofstream pcap_file( out_path, ios_base::out | ios_base::binary );
            in_stream.push( boost::iostreams::gzip_decompressor() );
            in_stream.push( gzip_file );
            boost::iostreams::copy( in_stream, pcap_file );
            gzip_file.close();
            fs::remove( out_path_gz );
            pcap_file.close();
            cout << "  done\n\n";

            // Decode messages.
            cout << "  decoding...\n";
            auto pcap_decoder = decoder();
            if( !pcap_decoder.open_file_for_decoding( out_path ) ) {
                cerr << "  failed to initialize pcap decoder.\n";
                std::abort();
            }
            const auto csv_path = data_directory / "csv" / ( std::get<0>( urls[idx] ) + ".csv" );
            std::ofstream csv_file( csv_path, ios_base::out );

            std::unique_ptr<iex_message_base> message;
            auto result = pcap_decoder.get_next_message( message );
            for( ; result == return_code_t::success; result = pcap_decoder.get_next_message( message ) ) {
                if( message->type() == message_type::trade_report ) {
                    const auto trade_report = dynamic_cast<trade_report_message*>( message.get() );
                    csv_file << trade_report->m_timestamp << ",";
                    csv_file << trade_report->m_symbol << ",";
                    csv_file << trade_report->m_price << ",";
                    csv_file << trade_report->m_size << endl;
                }
            }
            cout << "  done.\n\n";
            csv_file.close();
            fs::remove( out_path );
        }
    }

    size_t curl_write( void* contents, size_t size, size_t nmemb, std::string* s ) {
        const size_t new_length = size * nmemb;
        try {
            s->append( (char*)contents, new_length );
        } catch( std::bad_alloc& e ) {
            return 0;
        }
        return new_length;
    }

    vector<tuple<string, string>> downloader::get_urls() {
        auto curl = curl_easy_init();
        if( !curl ) {
            cerr << "Failed to initialize curl." << endl;
            std::terminate();
        }

        curl_easy_setopt( curl,
                          CURLOPT_URL,
                          hist_endpoint.c_str() );
        curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, curl_write );
        string response = "";
        curl_easy_setopt( curl, CURLOPT_WRITEDATA, &response );
        curl_easy_perform( curl );
        curl_easy_cleanup( curl );
        curl = nullptr;

        const auto js = json::parse( response );

        vector<tuple<string, string>> urls;
        for( auto& i : js ) {
            for( auto& j : i ) {
                if( j["feed"] == "TOPS" ) {
                    urls.emplace_back( std::make_tuple( j["date"], j["link"] ) );
                }
            }
        }
        for( auto i = urls.begin(); i != urls.end(); ++i ) {
            cout << std::get<0>( *i ) << endl;
            cout << std::get<1>( *i ) << endl;
        }

        return urls;
    }

}  // namespace jmnel::iex
