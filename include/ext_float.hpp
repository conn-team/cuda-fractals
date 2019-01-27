#pragma once

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

public:
    __both__ inline ExtFloat() {}
    __both__ inline ExtFloat(double val) { assign(val); }

    inline int exponent() const { return exp; }

    __both__ inline ExtFloat& operator=(double val) {
        assign(val);
        return *this;
    }

    __both__ inline explicit operator double() const {
        return abs(exp) < 1022 ? dealign(exp) : 0;
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

private:
    union {
        double fValue;
        uint64_t nValue;
    };
    int exp;
};
