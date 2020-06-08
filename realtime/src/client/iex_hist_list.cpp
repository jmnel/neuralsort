#include "iex_hist_list.hpp"

#include <boost/asio.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/stream_base.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/dynamic_body.hpp>
#include <boost/beast/http/message.hpp>
#include <boost/beast/http/string_body.hpp>
#include <boost/beast/http/verb.hpp>
#include <boost/beast/version.hpp>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;

namespace jmnel::iex {

    void get_trading_days() {

        const auto host = "iextrading.com";
        const auto port = "443";
        const auto target = "trading/market-data/";
        const auto version = 1.0;

        net::io_context ioc;
        ssl::context ctx( ssl::context::tlsv12_client );
        load_roo
    }

}  // namespace jmnel::iex
