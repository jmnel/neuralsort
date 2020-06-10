#pragma once

#include <Packet.h>
#include <PcapFileDevice.h>

#include <memory>

#include "message.hpp"

using std::string;

namespace jmnel::iex {

    enum class return_code_t {
        success = 0,
        class_not_initialized,
        failed_parsing_packet,
        failed_decoding_packet,
        unknown_message_type,
        end_of_stream
    };

    inline std::ostream& operator<<( std::ostream& os, const return_code_t code ) {

        switch( code ) {

            case return_code_t::success:
                os << "success";
                break;
            case return_code_t::class_not_initialized:
                os << "class not initialized";
                break;
            case return_code_t::failed_parsing_packet:
                os << "failed parsing packet";
                break;
            case return_code_t::failed_decoding_packet:
                os << "failed decoding packet";
                break;
            case return_code_t::unknown_message_type:
                os << "unknown message type";
                break;
            case return_code_t::end_of_stream:
                os << "end of stream";
                break;
        }

        return os;
    }

    class decoder {

    private:
        // The packet header length is 40.
        constexpr static size_t first_block_start = 40;

        iex_tp_header m_first_header;
        iex_tp_header m_last_decoded_header;

        std::unique_ptr<pcpp::IFileReaderDevice> m_reader;

        pcpp::Packet m_parsed_packet;

        const uint8_t* m_packet = nullptr;

        size_t m_block_offset = first_block_start;
        size_t m_packet_length = 0;

    public:
        decoder() = default;
        virtual ~decoder();

        bool open_file_for_decoding( string const& filename );
        return_code_t get_next_message( std::unique_ptr<iex_message_base>& message_ptr );

        inline iex_tp_header const& get_first_header() {
            return m_first_header;
        }

        inline iex_tp_header const& get_last_decoded_header() {
            return m_last_decoded_header;
        }

    private:
        return_code_t parse_next_packet( iex_tp_header& header );

        inline uint16_t get_block_size( const uint8_t* data ) {
            return *( reinterpret_cast<const uint16_t*>( data ) );
        }

        inline const uint8_t* get_block_data( const uint8_t* data ) {
            return data + 2;
        }
    };

}  // namespace jmnel::iex
