#pragma once
#include <boost/multiprecision/mpfr.hpp>

using namespace boost::multiprecision;

using BigFloat = mpfr_float;
using BigComplex = std::complex<BigFloat>;
