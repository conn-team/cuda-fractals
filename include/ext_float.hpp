#pragma once

#include "bignum.hpp"

// Double with extended exponent
class ExtFloat {
private:
    static constexpr int PRECISION = 256;

    __both__ inline void align() {
        exp += (int(nValue >> 52) & 0x7FF) - 1023;
        nValue = (nValue & 0x800FFFFFFFFFFFFFLL) | 0x3FF0000000000000LL;
    }

    __both__ inline double dealign(int exp) const {
        union {
            uint64_t n;
            double f;
        };
        n = nValue + (uint64_t(exp) << 52);
        return f;
    }

    __both__ inline void assign(double val) {
        fValue = val;
        exp = 0;
        align();
    }

    __both__ inline void assign(const BigFloat& val) {
        fValue = double(frexp(val, &exp));
        align();
    }

public:
    __both__ inline ExtFloat() {}
    __both__ inline ExtFloat(double val) { assign(val); }
    __both__ inline ExtFloat(const BigFloat& val) { assign(val); }

    __both__ static ExtFloat zero() {
        ExtFloat ret;
        ret.fValue = 1.f;
        ret.exp = -100000;
        return ret;
    }

    inline int exponent() const { return exp; }

    __both__ inline ExtFloat& operator=(double val) {
        assign(val);
        return *this;
    }

    __both__ inline ExtFloat& operator=(const BigFloat& val) {
        assign(val);
        return *this;
    }

    __both__ inline explicit operator double() const {
        return abs(exp) < 1022 ? dealign(exp) : 0;
    }

    __both__ inline explicit operator float() const {
        return abs(exp) < 126 ? float(dealign(exp)) : 0;
    }

    __both__ inline ExtFloat& operator+=(const ExtFloat& r) {
        int diff = r.exp - exp;
        if (diff > PRECISION) {
            *this = r;
        } else if (diff >= -PRECISION) {
            fValue += r.dealign(diff);
            align();
        }
        return *this;
    }

    __both__ inline ExtFloat& operator-=(const ExtFloat& r) {
        int diff = r.exp - exp;
        if (diff > PRECISION) {
            *this = -r;
        } else if (diff >= -PRECISION) {
            fValue -= r.dealign(diff);
            align();
        }
        return *this;
    }

    __both__ inline ExtFloat& operator*=(const ExtFloat& r) {
        fValue *= r.fValue;
        exp += r.exp;
        align();
        return *this;
    }

    __both__ inline ExtFloat& operator/=(const ExtFloat& r) {
        fValue /= r.fValue;
        exp -= r.exp;
        align();
        return *this;
    }

    __both__ inline ExtFloat operator+(const ExtFloat& r) const {
        ExtFloat tmp = *this;
        tmp += r;
        return tmp;
    }

    __both__ inline ExtFloat operator-(const ExtFloat& r) const {
        ExtFloat tmp = *this;
        tmp -= r;
        return tmp;
    }
    
    __both__ inline ExtFloat operator*(const ExtFloat& r) const {
        ExtFloat tmp = *this;
        tmp *= r;
        return tmp;
    }
    
    __both__ inline ExtFloat operator/(const ExtFloat& r) const {
        ExtFloat tmp = *this;
        tmp /= r;
        return tmp;
    }

    __both__ inline ExtFloat operator-() const {
        ExtFloat tmp = *this;
        tmp.fValue = -tmp.fValue;
        return tmp;
    }

    __both__ inline bool operator<(const ExtFloat& r) const {
        if (fValue > 0) {
            return r.fValue > 0 && (exp < r.exp || (exp == r.exp && fValue < r.fValue));
        } else if (fValue < 0) {
            return r.fValue >= 0 || exp > r.exp || (exp == r.exp && fValue < r.fValue);
        }
        return r.fValue > 0;
    }

    __both__ inline bool operator==(const ExtFloat& r) const {
        return exp == r.exp && fValue == r.fValue;
    }

    __both__ inline bool operator!=(const ExtFloat& r) const {
        return exp != r.exp || fValue != r.fValue;
    }

    __both__ inline bool operator>(const ExtFloat& r)  const { return r < *this; }
    __both__ inline bool operator<=(const ExtFloat& r) const { return *this < r || *this == r; }
    __both__ inline bool operator>=(const ExtFloat& r) const { return r <= *this; }

private:
    union {
        double fValue;
        uint64_t nValue;
    };
    int exp;
};
