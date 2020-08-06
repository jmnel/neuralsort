#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <new>
#include <sstream>
#include <string>
#include <tuple>

#include <curl/curl.h>
#include <curl/easy.h>
#include <fmt/core.h>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <nlohmann/json.hpp>

#include "common.hpp"
#include "debug/assert.hpp"
#include "decoder.hpp"
#include "downloader.hpp"
#include "message.hpp"

namespace fs = std::filesystem;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios_base;
using std::map;
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
        //        std::terminate();
        m_is_stopping = true;
    }

    // -- start_download function --
    void downloader::do_download() {

        // Get list of days, and associated endpoint urls.
        const auto urls = get_urls();

        // Loop through each day discoverd by previous query.
        for( size_t idx = 0; idx < urls.size(); ++idx ) {

            // Check first for stop signal.
            if( m_is_stopping ) {
                cout << "exit signal received. stopping." << endl;
                break;
            }

            // Construct filenames by appending extensions.
            const auto [name, endpoint] = urls[idx];
            const auto name_gz = name + ".pcap.gz";
            const auto name_pcap = name + ".pcap";
            const auto name_csv = name + ".csv";

            // Set paths of gz, pcap, and csv file.
            const auto path_gz = data_directory / "raw" / name_gz;
            const auto path_pcap = data_directory / "raw" / name_pcap;
            const auto path_csv = data_directory / "csv" / name_csv;

            // Check if data is already downloaded.
            if( fs::exists( path_csv ) ) {
                fmt::print( "  {} already downloaded.\n", name );
                continue;
            }

            fmt::print( "Downloading {} of {}: {}...\n",
                        idx + 1, urls.size(), name );

            // Try to initialize curl context.
            auto curl = curl_easy_init();
            if( !curl ) {
                cerr << "  Failed to initialize curl.\n";
                std::terminate();
            }

            // Delete gz archive if it already exists.
            if( fs::exists( path_gz ) ) {
                fmt::print( "  {} exists, deleting.\n", name_gz );
                fs::remove( path_gz );
            }
            // Delete pcap file if it already exists.
            if( fs::exists( path_pcap ) ) {
                fmt::print( "  {} exists, deleting.\n", name_pcap );
                fs::remove( path_pcap );
            }

            // Open file handle for download of gz archive.
            FILE* download_file_handle = nullptr;
            if( download_file_handle = fopen( path_gz.c_str(), "w" ); !download_file_handle ) {
                cerr << "  Failed to create file for download." << endl;
                std::abort();
            }

            // Setup curl with download endpoint url, and set output to file.
            curl_easy_setopt( curl,
                              CURLOPT_URL,
                              endpoint.c_str() );
            string response = "";
            curl_easy_setopt( curl, CURLOPT_WRITEDATA, download_file_handle );

            // Download the file.
            curl_easy_perform( curl );

            // Perform cleanup.
            curl_easy_cleanup( curl );
            curl = nullptr;
            fclose( download_file_handle );
            cout << "  Download complete.\n";

            // Uncompress downloaded gzip file with stream filter.
            cout << "  Uncompressing...\n";
            std::ifstream gzip_file( path_gz, ios_base::in | ios_base::binary );
            filtering_streambuf<boost::iostreams::input> in_stream;
            std::ofstream pcap_file( path_pcap, ios_base::out | ios_base::binary );
            in_stream.push( boost::iostreams::gzip_decompressor() );
            in_stream.push( gzip_file );
            boost::iostreams::copy( in_stream, pcap_file );

            // Perform cleanup.
            gzip_file.close();
            fs::remove( path_gz );
            pcap_file.close();
            cout << "  Done\n\n";

            // Decode messages.
            cout << "  Decoding...\n";
            auto pcap_decoder = decoder();
            if( !pcap_decoder.open_file_for_decoding( path_pcap ) ) {
                cerr << "  Failed to initialize pcap decoder.\n";
                std::abort();
            }
            std::ofstream csv_file( path_csv, ios_base::out );

            // Get header message.
            std::unique_ptr<iex_message_base> message;
            auto result = pcap_decoder.get_next_message( message );

            // Decode all remaining messages in pcap dump.
            size_t count = 0;
            for( ; result == return_code_t::success; result = pcap_decoder.get_next_message( message ) ) {
                if( message->type() == message_type::trade_report ) {
                    const auto trade_report = dynamic_cast<trade_report_message*>( message.get() );
                    csv_file << trade_report->m_timestamp << ",";
                    csv_file << trade_report->m_symbol << ",";
                    csv_file << trade_report->m_price << ",";
                    csv_file << trade_report->m_size << endl;
                    count++;
                }
            }
            fmt::print( "  Done, found {} trade execution messages.\n\n", count );
            csv_file.close();
            fs::remove( path_pcap );
        }
    }

    // -- curl_write function --
    size_t curl_write( void* contents, size_t size, size_t nmemb, std::string* s ) {
        const size_t new_length = size * nmemb;
        try {
            s->append( (char*)contents, new_length );
        } catch( std::bad_alloc& e ) {
            return 0;
        }
        return new_length;
    }

    // -- get_urls function --
    vector<tuple<string, string>> downloader::get_urls() {

        // Initialize curl.
        auto curl = curl_easy_init();
        if( !curl ) {
            cerr << "Failed to initialize curl." << endl;
            std::terminate();
        }
        curl_easy_setopt( curl,
                          CURLOPT_URL,
                          hist_endpoint.c_str() );
        curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, curl_write );

        // Download day listing from endpoint.
        string response = "";
        curl_easy_setopt( curl, CURLOPT_WRITEDATA, &response );
        curl_easy_perform( curl );
        curl_easy_cleanup( curl );
        curl = nullptr;

        // Extract JSON for endpoint item attributes.
        const auto js = json::parse( response );
        map<string, vector<endpoint_item>> items;
        for( auto& i : js ) {
            for( auto& j : i ) {
                if( j["feed"] == "TOPS" ) {
                    const auto date = j["date"].get<std::string>();
                    const auto link = j["link"].get<std::string>();
                    const auto protocol = j["protocol"].get<std::string>();
                    const auto size = std::stoull( j["size"].get<std::string>() );
                    const auto version = j["version"];

                    if( items.count( date ) < 1 ) {
                        items.insert( { date, {} } );
                    }
                    items[date].emplace_back(
                        endpoint_item{ date,
                                       link,
                                       protocol,
                                       size,
                                       version } );
                }
            }
        }

        // Some items return multiple links. Select TOPS v1.6 if both are present.
        vector<tuple<string, string>> urls;
        for( auto it = items.begin(); it != items.end(); ++it ) {
            auto&& [key, item] = *it;
            if( item.size() > 1 ) {
                for( auto q : item ) {
                    fmt::print( "  {}, {}\n", q.version, q.size );
                }
                assertf( item.size() == 2 );
                assertf( item[0].version == "1.5" );
                assertf( item[1].version == "1.6" );
                assertf( item[1].size > 1e3 );

                urls.emplace_back( std::make_tuple( item[1].date, item[1].link ) );
            } else {
                urls.emplace_back( std::make_tuple( item[0].date, item[0].link ) );
            }
        }

        std::reverse( urls.begin(), urls.end() );

        return urls;
    }

}  // namespace jmnel::iex
