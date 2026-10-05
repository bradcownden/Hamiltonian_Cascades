// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// High-precision floating-point type (Boost.Multiprecision float128, or
// cpp_bin_float_100 where __float128 is unavailable), math overloads and
// random generators. Boost is distributed under the Boost Software License 1.0.
// See README.md and LICENSE for details.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include "boost/math/policies/error_handling.hpp"

#if defined(BOOST_MATH_HAS_NVRTC) || defined(BOOST_MATH_NO_EXCEPTIONS)
namespace boost {
namespace math {
class rounding_error : public std::runtime_error {
public:
    explicit rounding_error(const std::string& s) : std::runtime_error(s) {}
};
} // namespace math
} // namespace boost
#endif

#include "boost/multiprecision/cpp_bin_float.hpp"
#include "boost/multiprecision/float128.hpp"

// Detect availability of __float128 — GCC on x86_64 Linux/WSL2 generally has it while Clang on macOS (especially arm64) generally doesn't. Use Boost's cpp_bin_float_100 as a fallback for high precision on platforms without native __float128 support.
#if defined(__GNUC__) && !defined(__clang__) && defined(__SIZEOF_FLOAT128__)
      using bigfloat = boost::multiprecision::float128;
#else
      using bigfloat = boost::multiprecision::cpp_bin_float_100;
#endif

using boost::math::constants::pi;
using namespace boost::math;

/*************************************
 ********** BOOST CONSTANTS **********
 *************************************/

const bigfloat zero = static_cast<bigfloat>(0);
const bigfloat one = static_cast<bigfloat>(1);
const bigfloat two = static_cast<bigfloat>(2);
const bigfloat three = static_cast<bigfloat>(3);
const bigfloat four = static_cast<bigfloat>(4);
const bigfloat B_PI = pi<bigfloat>();

/*************************************
 ********** BOOST SHORTHAND **********
 *************************************/

inline bigfloat blog(const bigfloat& x) {
    return boost::multiprecision::log(x);
};
inline bigfloat bsqrt(const bigfloat& x) {
    return boost::multiprecision::sqrt(x);
};
inline bigfloat bpow(const bigfloat& x, const bigfloat& y) {
    return boost::multiprecision::pow(x, y);
};
inline bigfloat bcos(const bigfloat& x) {
    return boost::multiprecision::cos(x);
};
inline bigfloat bsin(const bigfloat& x) {
    return boost::multiprecision::sin(x);
};
inline bigfloat bexp(const bigfloat& x) {
    return boost::multiprecision::exp(x);
};

/*****************************************************
 ****************** MATH OVERLOADS  ******************
 *****************************************************/

namespace bmath {
    // ==== Pow overload ====
    template<typename T1, typename T2>
    inline bigfloat Pow(T1 x, T2 y) {
        // Promote both arguments to bigfloat if possible
        static_assert(std::is_convertible_v<T1, bigfloat>,
                    "Pow base must be convertible to bigfloat");
        static_assert(std::is_convertible_v<T2, bigfloat>,
                    "Pow exponent must be convertible to bigfloat");

        return bpow(static_cast<bigfloat>(x),
                                        static_cast<bigfloat>(y));
    }

    // ==== Sqrt overload ====
    template<typename T1>
    inline bigfloat Sqrt(T1 x) {
        // Promote both arguments to bigfloat if possible
        static_assert(std::is_convertible_v<T1, bigfloat>,
                    "Sqrt argument must be convertible to bigfloat");

        return bsqrt(static_cast<bigfloat>(x));
    }

    // ==== Cos overload ====
    template<typename T1>
    inline bigfloat cos(T1 x) {
        static_assert(std::is_convertible_v<T1, bigfloat>,
                    "cos argument must be convertible to bigfloat");
        return bcos(static_cast<bigfloat>(x));
    }

    // ==== Sin overload ====
    template<typename T1>
    inline bigfloat sin(T1 x) {
        static_assert(std::is_convertible_v<T1, bigfloat>,
                    "sin argument must be convertible to bigfloat");
        return bsin(static_cast<bigfloat>(x));
    }

    // ==== Exp overload ====
    template<typename T1>
    inline bigfloat exp(T1 x) {
        static_assert(std::is_convertible_v<T1, bigfloat>,
                    "exp argument must be convertible to bigfloat");
        return bexp(static_cast<bigfloat>(x));
    }
}

/*************************************
 ********* RANDOM GENERATORS *********
 *************************************/

// Uniform bigfloat in [0, 1) with full mantissa precision.
// Sums multiple 64-bit draws as fractional bits:
//   float128         : 113-bit mantissa => 2 draws suffice
//   cpp_bin_float_100: ~333-bit mantissa => 6 draws suffice
inline bigfloat uniform_bigfloat(std::mt19937_64& rng)
{
    constexpr int DRAWS = 6;
    const bigfloat inv64 = boost::multiprecision::ldexp(bigfloat(1), -64);
    bigfloat result = bigfloat(0);
    bigfloat factor = inv64;
    for (int i = 0; i < DRAWS; ++i) {
        result += static_cast<bigfloat>(rng()) * factor;
        factor *= inv64;
    }
    return result;
}

// Gaussian bigfloat with mean 0 and variance 1.
// Uses Box-Muller and reuses one generated normal sample for efficiency.
inline bigfloat gaussian_bigfloat(std::mt19937_64& rng)
{
    static thread_local bool has_spare = false;
    static thread_local bigfloat spare = bigfloat(0);

    if (has_spare) {
        has_spare = false;
        return spare;
    }

    bigfloat u1 = uniform_bigfloat(rng);
    while (u1 <= zero) {
        u1 = uniform_bigfloat(rng);
    }
    const bigfloat u2 = uniform_bigfloat(rng);

    const bigfloat radius = bsqrt(-two * blog(u1));
    const bigfloat theta = two * B_PI * u2;

    spare = radius * bsin(theta);
    has_spare = true;
    return radius * bcos(theta);
}

// Convenience overload using a thread-local seeded RNG.
inline bigfloat gaussian_bigfloat()
{
    static thread_local std::mt19937_64 rng(
        []() {
            std::seed_seq seq{
                std::random_device{}(), std::random_device{}(),
                std::random_device{}(), std::random_device{}()
            };
            return std::mt19937_64(seq);
        }()
    );
    return gaussian_bigfloat(rng);
}

// Convenience overload using a thread-local seeded RNG.
inline bigfloat uniform_bigfloat()
{
    static thread_local std::mt19937_64 rng(
        []() {
            std::seed_seq seq{
                std::random_device{}(), std::random_device{}(),
                std::random_device{}(), std::random_device{}()
            };
            return std::mt19937_64(seq);
        }()
    );
    return uniform_bigfloat(rng);
}


