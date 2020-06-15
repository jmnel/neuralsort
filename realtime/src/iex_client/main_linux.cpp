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

    return EXIT_SUCCESS;
}
