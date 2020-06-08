#include <iostream>
#include <string>
#include <unordered_set>

#include "iex_decoder.hpp"
#include "iex_hist_list.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

using namespace jmnel;

int main( int, char *[] ) {
    cout << "Client started." << endl;

    iex::get_trading_days();

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

    //    for( ; result == iex::return_code_t::success; result = decoder.get_next_message( message ) ) {

    //        if( message->type() == iex::message_type::trade_report ) {
    //            const auto trade_report = dynamic_cast<iex::trade_report_message *>( message.get() );

    //            symbols.insert( trade_report->m_symbol );

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
