//#include <SQLiteCpp/SQLiteCpp.h>

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

//#include "SQLiteCpp/Database.h"
#include "idownloader.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

using namespace jmnel;

//SQLite::Database prepare_database() {
//    cout << "Preparing database..." << endl;
//    SQLite::Database db( "../data/iex.sqlite3", SQLite::OPEN_CREATE | SQLite::OPEN_READWRITE );

//    db.exec( "PRAGMA foreign_keys = ON;" );
//    db.exec( R"(
//DROP TABLE IF EXISTS iex_symbols;
//    )" );
//    db.exec( R"(
//DROP TABLE IF EXISTS iex_trade_reports;
//    )" );
//    db.exec( R"(
//CREATE TABLE iex_trade_reports(
//    id INTEGER PRIMARY KEY,
//    timestamp INTEGER NOT NULL,
//    symbol CHAR(16) NOT NULL,
//    price INEGER NOT NULL,
//    size INTEGER NOT NULL);
//    )" );

//    return db;
//}

int main( int, char *[] ) {
    iex::idownloader::run();

    //    const auto dow

    //    auto db = prepare_database();

    //    const auto file_path = string( "/home/jacques/repos/jmnel/neuralsort/realtime/data/data_feeds_20180127_20180127_IEXTP1_TOPS1.6.pcap" );
    //    const auto file_path = string( "/home/jacques/repos/jmnel/neuralsort/realtime/data/data_feeds_20200604_20200604_IEXTP1_TOPS1.6.pcap" );
    //    auto decoder = iex::decoder();
    //    if( !decoder.open_file_for_decoding( file_path ) ) {
    //        cerr << "Failed to initalize decoder." << endl;
    //        std::terminate();
    //    }

    //    std::unique_ptr<iex::iex_message_base> message;
    //    auto result = decoder.get_next_message( message );

    //    auto symbols = std::unordered_set<string>();
    //    auto message_count = 0;

    //    vector<iex::trade_report_message> messages;

    //    for( ; result == iex::return_code_t::success; result = decoder.get_next_message( message ) ) {

    //        if( message->type() == iex::message_type::trade_report ) {
    //            const auto trade_report = dynamic_cast<iex::trade_report_message *>( message.get() );
    //                        messages.emplace_back( trade_report );

    //                        symbols.insert( trade_report->m_symbol );

    //            cout << trade_report->m_timestamp << ",";
    //            cout << trade_report->m_symbol << ",";
    //            cout << trade_report->m_price << ",";
    //            cout << trade_report->m_size << endl;

    //            cout << *trade_report;
    //            cout << endl;

    //            message_count++;
    //        }
    //    }

    //    cout << "Symbols found:" << endl;
    //    for( const auto &s : symbols ) {
    //        cout << "  " << s << endl;
    //    }

    //    cout << "num message: " << message_count << endl;

    return EXIT_SUCCESS;
}
