// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Minimal complex-number type built on bigfloat.
// See README.md and LICENSE for details.

#pragma once
#include "bigfloat.hpp" // bigfloat type

/***************************************************
 ********** COMPLEX HIGH PRECISION STRUCT **********
 ***************************************************/

struct complex128 {
    bigfloat re;
    bigfloat im;

    complex128(bigfloat r = static_cast<bigfloat>(0),
               bigfloat i = static_cast<bigfloat>(0))
        : re(r), im(i) {}

    // Complex/complex arithmetic
    complex128 operator+(const complex128& z) const {
        return {re + z.re, im + z.im};
    }
    complex128 operator-(const complex128& z) const {
        return {re - z.re, im - z.im};
    }
    complex128 operator*(const complex128& z) const {
        return {re * z.re - im * z.im, re * z.im + im * z.re};
    }
    complex128 operator/(const complex128& z) const {
        bigfloat denom = z.re * z.re + z.im * z.im;
        return {(re * z.re + im * z.im) / denom,
                (im * z.re - re * z.im) / denom};
    }
    bool operator==(const complex128& z) const {
        return re == z.re && im == z.im;
    }

    complex128 cexp() const {
        bigfloat exp_real = boost::multiprecision::exp(re);
        return {exp_real * bcos(im),
                exp_real * bsin(im)};
    }

    // Complex/real arithmetic
    complex128 operator*(const bigfloat& scalar) const {
        return {re * scalar, im * scalar};
    }
    friend complex128 operator*(const bigfloat& scalar, const complex128& z) {
        return {scalar * z.re, scalar * z.im};
    }
    complex128 operator+(const bigfloat& scalar) const {
        return {re + scalar, im};
    }
    friend complex128 operator+(const bigfloat& scalar, const complex128& z) {
        return {scalar + z.re, z.im};
    }
    complex128 operator-(const bigfloat& scalar) const {
        return {re - scalar, im};
    }
    friend complex128 operator-(const bigfloat& scalar, const complex128& z) {
        return {scalar - z.re, z.im};
    }
    complex128 operator/(const bigfloat& scalar) const {
        return {re / scalar, im / scalar};
    }
    friend complex128 operator/(const bigfloat& scalar, const complex128& z) {
        bigfloat denom = z.re * z.re + z.im * z.im;
        return {scalar * z.re / denom, -scalar * z.im / denom};
    }

    // Utilities
    complex128 conjugate() const {
        return {re, -im};
    }
    bigfloat normsq() const {
        return re * re + im * im;
    }
    bigfloat real() const {
        return re;
    }
    bigfloat imag() const {
        return im;
    }
    complex128 maxabs(const complex128& z) const {
        bigfloat norm = re*re + im*im;
        bigfloat znorm = z.normsq();
        if (norm >= znorm){
            return {re, im};
        }
        else
        {
            return z;
        }
    }
};

// +/- imaginary
const complex128 im(zero,one);
const complex128 mim(zero,-one);