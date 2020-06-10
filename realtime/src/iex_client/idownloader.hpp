#pragma once

#include <memory>

namespace jmnel::iex {

    class idownloader {
    protected:
        static std::unique_ptr<idownloader> singleton;

    public:
        virtual ~idownloader() = default;
        virtual void request_stop() = 0;
        virtual void start_download() = 0;

        static bool run();
        static bool stop();
        static bool is_running();
    };

}  // namespace jmnel::iex
