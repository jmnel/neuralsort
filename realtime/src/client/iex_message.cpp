#include "iex_message.hpp"

#include <algorithm>
#include <iostream>

using std::cerr;

namespace jmnel::iex {

    template <typename T>
    T get_numeric( const uint8_t* data, const size_t offset ) {
        return *( reinterpret_cast<const T*>( &data[offset] ) );
    }

    double get_price( const uint8_t* data, const size_t offset ) {
        return *( reinterpret_cast<const int64_t*>( &data[offset] ) ) / 10000.0;
    }

    string get_string( const uint8_t* data, const size_t offset, const size_t length ) {

        string s = string( ( reinterpret_cast<const char*>( &data[offset] ) ), length );

        // Remove excess whitespace.
        s.erase(
            std::find_if( s.rbegin(), s.rend(), []( int ch ) { return !std::isspace( ch ); } ).base(), s.end() );

        return s;
    }

    bool validate_timestamp( const int64_t timestamp ) {
        return ( timestamp > 1382659200000000000 ) && ( timestamp < 4102444800000000000 );
    }

    bool iex_tp_header::decode( const uint8_t* data ) {
        m_version = get_numeric<uint8_t>( data, 0 );
        m_protocol_id = get_numeric<uint16_t>( data, 2 );
        m_channel_id = get_numeric<uint32_t>( data, 4 );
        m_session_id = get_numeric<uint16_t>( data, 8 );
        m_payload_len = get_numeric<uint16_t>( data, 12 );
        m_message_count = get_numeric<uint16_t>( data, 14 );
        m_stream_offset = get_numeric<uint64_t>( data, 16 );
        m_first_msg_sq_num = get_numeric<uint64_t>( data, 24 );
        m_send_time = get_numeric<uint64_t>( data, 32 );

        if( m_version != 1 ) {
            cerr << "Error: Transport protocol version has changed." << endl;
            return false;
        }

        return true;
    }

    bool system_event_message::decode( const uint8_t* data ) {
        m_system_event = static_cast<system_event_message::code>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );

        return validate_timestamp( m_timestamp );
    }

    bool security_directory_message::decode( const uint8_t* data ) {
        m_flags = get_numeric<uint8_t>( data, 1 );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_round_lot_size = get_numeric<uint32_t>( data, 18 );
        m_adjusted_poc_price = get_price( data, 22 );
        m_luld_tier = static_cast<luld_tier>( get_numeric<u_int8_t>( data, 30 ) );

        return validate_timestamp( m_timestamp );
    }

    bool trading_status_message::decode( const uint8_t* data ) {
        m_trading_status = static_cast<status>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_reason = get_string( data, 18, 4 );

        return validate_timestamp( m_timestamp );
    }

    bool operational_halt_status_message::decode( const uint8_t* data ) {
        m_operational_halt_status = static_cast<status>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );

        return validate_timestamp( m_timestamp );
    }

    bool short_sale_price_test_status_message::decode( const uint8_t* data ) {
        m_short_sale_test_in_effect = static_cast<bool>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_detail = static_cast<detail>( get_numeric<uint8_t>( data, 18 ) );

        return validate_timestamp( m_timestamp );
    }

    bool quote_update_message::decode( const uint8_t* data ) {
        m_flags = get_numeric<uint8_t>( data, 1 );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_bid_size = get_numeric<uint32_t>( data, 18 );
        m_bid_price = get_price( data, 22 );
        m_ask_size = get_numeric<uint32_t>( data, 38 );
        m_ask_price = get_price( data, 30 );

        return validate_timestamp( m_timestamp );
    }

    bool trade_report_message::decode( const uint8_t* data ) {
        m_flags = get_numeric<uint8_t>( data, 1 );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_size = get_numeric<uint32_t>( data, 18 );
        m_price = get_price( data, 22 );
        m_trade_id = get_numeric<uint64_t>( data, 30 );

        return validate_timestamp( m_timestamp );
    }

    bool official_price_message::decode( const uint8_t* data ) {
        m_price_type = static_cast<price_type>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_price = get_price( data, 18 );

        return validate_timestamp( m_timestamp );
    }

    bool auction_information_message::decode( const uint8_t* data ) {
        m_auction_type = static_cast<auction_type>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_paired_shares = get_numeric<uint32_t>( data, 18 );
        m_reference_price = get_price( data, 22 );
        m_indicative_clearing_price = get_price( data, 30 );
        m_imbalance_shares = get_numeric<uint32_t>( data, 38 );
        m_imbalance_side = static_cast<auction_information_message::imbalance_side>(
            get_numeric<uint8_t>( data, 42 ) );
        m_extension_number = get_numeric<uint8_t>( data, 43 );
        m_scheduled_auction_time = get_numeric<uint32_t>( data, 44 );
        m_auction_book_clearing_price = get_price( data, 48 );
        m_collar_reference_price = get_price( data, 56 );
        m_lower_auction_collar = get_price( data, 64 );
        m_upper_auction_collar = get_price( data, 72 );

        return validate_timestamp( m_timestamp );
    }

    bool price_level_update_message::decode( const uint8_t* data ) {
        m_flags = get_numeric<uint8_t>( data, 1 );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );
        m_size = get_numeric<uint32_t>( data, 18 );
        m_price = get_price( data, 22 );

        return validate_timestamp( m_timestamp );
    }

    bool security_event_message::decode( const uint8_t* data ) {
        m_event = static_cast<security_event_message::security_event_type>( get_numeric<uint8_t>( data, 1 ) );
        m_timestamp = get_numeric<uint64_t>( data, 2 );
        m_symbol = get_string( data, 10, 8 );

        return validate_timestamp( m_timestamp );
    }

    std::unique_ptr<iex_message_base> message_factory( const uint8_t* data ) {

        const auto type = static_cast<message_type>( *data );

        switch( type ) {
            case message_type::quote_update:
                return std::make_unique<quote_update_message>();

            case message_type::trading_status:
                return std::make_unique<trading_status_message>();

            case message_type::system_event:
                return std::make_unique<system_event_message>();

            case message_type::security_directory:
                return std::make_unique<security_directory_message>();

            case message_type::operational_halt_status:
                return std::make_unique<operational_halt_status_message>();

            case message_type::short_sale_price_test_status:
                return std::make_unique<short_sale_price_test_status_message>();

            case message_type::trade_report:

            case message_type::trade_break:
                return std::make_unique<trade_report_message>();

            case message_type::official_price:
                return std::make_unique<official_price_message>();

            case message_type::auction_information:
                return std::make_unique<auction_information_message>();

            case message_type::price_level_update_buy:
            case message_type::price_level_update_sell:
                return std::make_unique<price_level_update_message>( type );

            case message_type::security_event:
                return std::make_unique<security_event_message>( type );

            default:
                return nullptr;
        }
    }

    std::ostream& operator<<( std::ostream& os, iex_tp_header const& header ) {
        os << header.type() << endl;
        os << "  version: " << header.m_version << endl;
        os << "  protocol id: " << header.m_protocol_id << endl;
        os << "  channel id: " << header.m_channel_id << endl;
        os << "  session id: " << header.m_session_id << endl;
        os << "  payload length: " << header.m_payload_len << endl;
        os << "  message count: " << header.m_message_count << endl;
        os << "  stream offset: " << header.m_stream_offset << endl;
        os << "  first message sequence number: " << header.m_first_msg_sq_num << endl;
        os << "  send time: " << header.m_send_time << endl;

        return os;
    }

    std::ostream& operator<<( std::ostream& os, system_event_message const& msg ) {
        os << msg.type() << endl;
        os << "  code: " << msg.m_system_event << endl;

        return os;
    }

    std::ostream& operator<<( std::ostream& os, trading_status_message const& msg ) {
        os << msg.type() << endl;
        os << "  trading status: " << msg.m_trading_status << endl;
        os << "  symbol: " << msg.m_symbol << endl;
        os << "  reason: " << msg.m_reason << endl;

        return os;
    }

    std::ostream& operator<<( std::ostream& os, quote_update_message const& msg ) {
        os << msg.type() << endl;
        os << "  symbol: " << msg.m_symbol << endl;
        os << "  bid size: " << msg.m_bid_size << endl;
        os << "  bid price: " << msg.m_bid_price << endl;
        os << "  ask size: " << msg.m_ask_size << endl;
        os << "  ask price: " << msg.m_ask_price << endl;
        os << "  flags: " << msg.m_flags << endl;

        return os;
    }

    std::ostream& operator<<( std::ostream& os, trade_report_message const& msg ) {
        os << msg.type() << endl;
        os << "  symbol: " << msg.m_symbol << endl;
        os << "  price: " << msg.m_price << endl;
        os << "  size: " << msg.m_size << endl;
        os << "  trade id: " << msg.m_trade_id << endl;
        os << "  flags: " << msg.m_flags << endl;

        return os;
    }

    std::ostream& operator<<( std::ostream& os, official_price_message const& msg ) {
        os << msg.type() << endl;
        os << "  symbol: " << msg.m_symbol << endl;
        if( msg.m_price_type == official_price_message::price_type::open_price ) {
            os << "  price type: 0x51 open" << endl;
        } else {
            os << "  price type: 0x4d clsoe" << endl;
        }
        os << "  price: " << msg.m_price << endl;

        return os;
    }

}  // namespace jmnel::iex
