#include "iex_decoder.hpp"

#include <PayloadLayer.h>

#include <iomanip>
#include <iostream>

using std::cerr;
using std::cout;

namespace jmnel::iex {

    decoder::~decoder() {
        if( m_reader ) {
            m_reader->close();
        }
    }

    bool decoder::open_file_for_decoding( string const &filename ) {

        m_reader.reset( pcpp::IFileReaderDevice::getReader( filename.c_str() ) );

        // Check that reader was successfully created.
        if( !m_reader ) {
            cerr << "Failed to initialize Pcap reader." << endl;
            std::terminate();
        }

        // Open the reader.
        if( !m_reader->open() ) {
            cerr << "Failed to open file " << filename << endl;
            std::terminate();
        }

        if( parse_next_packet( m_first_header ) != return_code_t::success ) {
            cerr << "Failed to parse first packet." << endl;
            std::terminate();
        }

        if( m_packet_length <= first_block_start ) {
            m_packet = nullptr;
        }

        return true;
    }

    return_code_t decoder::parse_next_packet( iex_tp_header &header ) {

        if( !m_reader ) {
            cerr << "The decoder has not opened a file for reading." << endl;
            std::terminate();
            return return_code_t::class_not_initialized;
        }

        // Parse the packet.
        pcpp::RawPacket raw_packet;
        if( !m_reader->getNextPacket( raw_packet ) ) {
            cout << "Packet reader has no more packets to decode." << endl;
            return return_code_t::end_of_stream;
        }

        m_parsed_packet = pcpp::Packet( &raw_packet );

        // Extract the payload layer. We construct IEX message using payload.
        pcpp::PayloadLayer *payload_layer = m_parsed_packet.getLayerOfType<pcpp::PayloadLayer>();
        if( !payload_layer ) {
            cerr << "Generic payload layer not found for IEX message data packet." << endl;
            return return_code_t::failed_parsing_packet;
        }
        m_packet = payload_layer->getData();
        m_packet_length = payload_layer->getDataLen();
        m_block_offset = first_block_start;

        // Handle header packet.
        const auto success = header.decode( m_packet );
        if( !success ) {
            cerr << "Failed to decode header." << endl;
            return return_code_t::failed_decoding_packet;
        }

        return return_code_t::success;
    }

    return_code_t decoder::get_next_message( std::unique_ptr<iex_message_base> &message ) {

        if( !m_reader ) {
            cerr << "The decoder has not been properly initialized for file reading.";
            std::terminate();
            return return_code_t::class_not_initialized;
        }

        // Check if the packet iterator is valid. Parse next packet otherwise.
        if( !m_packet ) {

            do {
                // Parse next packet.
                const auto result = parse_next_packet( m_last_decoded_header );
                if( result != return_code_t::success ) {
                    return result;
                }

                // Some packets are empty. The server sends heartbeat packets every second, when there are no messages.
            } while( m_last_decoded_header.m_payload_len == 0 );
        }

        // Get a pointer to the current block.
        const uint8_t *block = m_packet + m_block_offset;

        // Get length of current block.
        const auto block_length = get_block_size( block );

        // Get data pointer within current block.
        const uint8_t *data = get_block_data( block );

        // Advance block offset to next block.
        m_block_offset += block_length + 2;

        // Reset pointer after end of whole packet.
        if( m_block_offset >= m_packet_length ) {
            m_packet = 0;
        }

        message = message_factory( data );
        if( !message ) {
            cerr << "Unkown message type: 0x" << std::hex << data << endl;
            cerr << "  Block length " << block_length << endl;
            std::terminate();
            return return_code_t::unknown_message_type;
        }

        if( !message->decode( data ) ) {
            std::cerr << "Failed decoding packet." << endl;
            return return_code_t::failed_decoding_packet;
        }

        return return_code_t::success;
    }

}  // namespace jmnel::iex
