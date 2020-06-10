#include "idownloader.hpp"

#include <csignal>
#include <iostream>
#include <memory>

#include "downloader.hpp"

using std::cerr;
using std::cout;
using std::endl;

namespace jmnel::iex {

    std::unique_ptr<idownloader> idownloader::singleton = nullptr;

    // -- handle_sigint function --
    void handle_sigint( int signum ) {
        cout << "Stop signal received." << endl;
        if( idownloader::is_running() ) {
            idownloader::stop();
        }
    }

    // -- run function --
    bool idownloader::run() {
        if( singleton ) {
            cerr << "Can't start downloader. Already running." << endl;
            return false;
        }

        signal( SIGINT, handle_sigint );
        singleton = std::make_unique<downloader>();
        singleton->start_download();

        return true;
    }

    // -- stop function --
    bool idownloader::stop() {
        if( !singleton ) {
            cerr << "Can't stop downloader. Not running." << endl;
            return false;
        }

        singleton->request_stop();
        singleton.release();
        return true;
    }

    // -- is_running function --
    bool idownloader::is_running() {
        return bool( singleton );
    }

}  // namespace jmnel::iex
