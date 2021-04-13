/**
Implementation of version.h

\file version.hpp
\copyright Copyright 2021. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef PRRNG_VERSION_HPP
#define PRRNG_VERSION_HPP

#include "version.h"

namespace prrng {

namespace detail {

    inline std::string unquote(const std::string& arg)
    {
        std::string ret = arg;
        ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
        return ret;
    }

}

inline std::string version()
{
    return detail::unquote(std::string(QUOTE(PRRNG_VERSION)));
}

inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("prrng=" + version());

    ret.push_back("xtensor=" +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

    return ret;
}

} // namespace prrng

#endif
