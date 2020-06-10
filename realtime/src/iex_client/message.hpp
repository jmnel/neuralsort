#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

using std::endl;
using std::string;

namespace jmnel::iex {

    enum class message_type {
        no_data = 0xff,
        stream_header = 0x00,
        system_event = 0x53,
        security_directory = 0x44,
        security_event = 0x45,
        trading_status = 0x48,
        operational_halt_status = 0x4f,
        short_sale_price_test_status = 0x50,
        quote_update = 0x51,
        trade_report = 0x54,
        official_price = 0x58,
        trade_break = 0x42,
        auction_information = 0x41,
        price_level_update_buy = 0x38,
        price_level_update_sell = 0x35
    };

    inline std::ostream& operator<<( std::ostream& os, const message_type msg_type ) {
        switch( msg_type ) {
            case message_type::no_data:
                os << "NoData";
                break;
            case message_type::stream_header:
                os << "StreamHeader";
                break;
            case message_type::system_event:
                os << "SystemEvent";
                break;
            case message_type::security_directory:
                os << "SecurityDirectory";
                break;
            case message_type::security_event:
                os << "SecurityEvent";
                break;
            case message_type::trading_status:
                os << "TradingStatus";
                break;
            case message_type::operational_halt_status:
                os << "OperationalHaltStatus";
                break;
            case message_type::short_sale_price_test_status:
                os << "ShortSalePriceTestStatus";
                break;
            case message_type::quote_update:
                os << "QuoteUpdate";
                break;
            case message_type::trade_report:
                os << "TradeReport";
                break;
            case message_type::official_price:
                os << "OfficialPrice";
                break;
            case message_type::trade_break:
                os << "TradeBreak";
                break;
            case message_type::auction_information:
                os << "AuctionInformation";
                break;
            case message_type::price_level_update_buy:
                os << "PriceLevelUpdateBuy";
                break;
            case message_type::price_level_update_sell:
                os << "PriceLevelSell";
                break;
        }
        return os;
    }

    class iex_message_base {
    protected:
        message_type m_type = message_type::no_data;

    public:
        uint64_t m_timestamp;

        virtual ~iex_message_base() = default;

        virtual bool decode( const uint8_t* data ) = 0;

        message_type type() const {
            return m_type;
        }
    };

    struct iex_tp_header : public iex_message_base {

        uint8_t m_version;
        uint16_t m_protocol_id;
        uint32_t m_channel_id;
        uint32_t m_session_id;
        uint16_t m_payload_len;
        uint16_t m_message_count;
        int64_t m_stream_offset;
        int64_t m_first_msg_sq_num;

        int64_t m_send_time;

        iex_tp_header() {
            m_type = message_type::stream_header;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct system_event_message : public iex_message_base {
        enum class code {
            start_of_messages = 0x4f,
            start_of_system_hours = 0x53,
            start_of_regular_market_hours = 0x52,
            end_of_regular_market_hours = 0x4d,
            end_of_system_hours = 0x45,
            end_of_messages = 0x43
        };

        code m_system_event;

        system_event_message() {
            m_type = message_type::system_event;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    inline std::ostream& operator<<( std::ostream& os, const system_event_message::code system_event ) {
        switch( system_event ) {
            case system_event_message::code::start_of_messages:
                os << "0x4f start of messages";
                break;
            case system_event_message::code::start_of_system_hours:
                os << "0x53 start of system hours";
                break;
            case system_event_message::code::start_of_regular_market_hours:
                os << "0x52 start of regular market hours";
                break;
            case system_event_message::code::end_of_regular_market_hours:
                os << "0x4d end of regular market hours";
                break;
            case system_event_message::code::end_of_system_hours:
                os << "0x45 end of system hours";
                break;
            case system_event_message::code::end_of_messages:
                os << "0x43 end of messages";
                break;
        }
        return os;
    }

    struct security_directory_message : public iex_message_base {
        enum class luld_tier {
            not_applicable = 0x0,
            tier_1_nm_stock = 0x1,
            tier_2_nm_stock = 0x2
        };

        uint8_t m_flags;
        string m_symbol;

        int m_round_lot_size;
        double m_adjusted_poc_price;

        luld_tier m_luld_tier;

        security_directory_message() {
            m_type = message_type::security_directory;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct trading_status_message : public iex_message_base {
        enum class status {
            trading_halted = 0x48,
            trading_halt_released = 0x4f,
            trading_paused = 0x50,
            trading = 0x54
        };

        status m_trading_status;
        string m_symbol;
        string m_reason;

        // Constructor
        trading_status_message() {
            m_type = message_type::trading_status;
        };

        virtual bool decode( const uint8_t* data ) final;
    };

    inline std::ostream& operator<<( std::ostream& os, const trading_status_message::status status ) {
        switch( status ) {

            case trading_status_message::status::trading_halted:
                os << "0x48 trading halted";
                break;

            case trading_status_message::status::trading_halt_released:
                os << "0x4f trading halt released";
                break;

            case trading_status_message::status::trading_paused:
                os << "0x50 trading paused";
                break;

            case trading_status_message::status::trading:
                os << "0x54 trading";
                break;
        }

        return os;
    }

    struct operational_halt_status_message : public iex_message_base {
        enum class status {
            operational_halt = 0x4f,
            not_halted = 0x4e
        };

        status m_operational_halt_status;
        string m_symbol;

        operational_halt_status_message() {
            m_type = message_type::operational_halt_status;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct short_sale_price_test_status_message : public iex_message_base {
        enum class detail {
            no_price_test = 0x20,
            short_sale_test_intraday_price_drop = 0x41,
            short_sale_test_continued = 0x43,
            short_sale_price_deactivated = 0x44,
            detail_not_available = 0x4e
        };

        bool m_short_sale_test_in_effect;
        string m_symbol;
        detail m_detail;

        short_sale_price_test_status_message() {
            m_type = message_type::short_sale_price_test_status;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct quote_update_message : public iex_message_base {

        uint8_t m_flags;
        string m_symbol;
        int m_bid_size;
        int m_ask_size;
        double m_bid_price;
        double m_ask_price;

        quote_update_message() {
            m_type = message_type::quote_update;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct trade_report_message : public iex_message_base {

        uint8_t m_flags;
        string m_symbol;
        int m_size;
        double m_price;
        int m_trade_id;

        trade_report_message() {
            m_type = message_type::trade_report;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct official_price_message : public iex_message_base {
        enum class price_type {
            open_price = 0x51,
            close_price = 0x4d
        };

        price_type m_price_type;
        string m_symbol;
        double m_price;

        official_price_message() {
            m_type = message_type::official_price;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct auction_information_message : public iex_message_base {
        enum class auction_type {
            opening_auction = 0x4f,
            closing_auction = 0x43,
            ipo_auction = 0x49,
            halt_auction = 0x48,
            volatility_auction = 0x56
        };

        enum class imbalance_side {
            buy_side_imbalance = 0x42,
            sell_side_imbalance = 0x53,
            no_imbalance = 0x4e
        };

        auction_type m_auction_type;
        string m_symbol;
        int m_paired_shares;
        double m_reference_price;
        double m_indicative_clearing_price;
        int m_imbalance_shares;
        imbalance_side m_imbalance_side;
        int m_extension_number;
        int m_scheduled_auction_time;
        double m_auction_book_clearing_price;
        double m_collar_reference_price;
        double m_lower_auction_collar;
        double m_upper_auction_collar;

        auction_information_message() {
            m_type = message_type::auction_information;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    struct price_level_update_message : public iex_message_base {

        uint8_t m_flags;
        string m_symbol;
        int m_size;
        int m_price;

        price_level_update_message( const message_type msg_type ) {
            m_type = msg_type;
        }

        virtual bool decode( const uint8_t* data );
    };

    struct security_event_message : public iex_message_base {

        enum class security_event_type {
            opening_process_complete = 0x4f,
            closing_process_complete = 0x43
        };

        security_event_type m_event;
        string m_symbol;

        security_event_message( const message_type msg_type ) {
            m_type = msg_type;
        }

        virtual bool decode( const uint8_t* data ) final;
    };

    std::unique_ptr<iex_message_base> message_factory( const uint8_t* data );

    // Declare string output stream operators for various messages.
    std::ostream& operator<<( std::ostream& os, iex_tp_header const& header );
    std::ostream& operator<<( std::ostream& os, system_event_message const& msg );
    std::ostream& operator<<( std::ostream& os, trading_status_message const& msg );
    std::ostream& operator<<( std::ostream& os, quote_update_message const& msg );
    std::ostream& operator<<( std::ostream& os, trade_report_message const& msg );
    std::ostream& operator<<( std::ostream& os, official_price_message const& msg );

}  // namespace jmnel::iex
