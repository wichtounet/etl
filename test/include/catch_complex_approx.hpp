//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "internal/catch_tostring.h"

#include <cmath>
#include <limits>

template<typename T>
struct ComplexApprox {
    explicit ComplexApprox(std::complex<T> value) : eps(std::numeric_limits<float>::epsilon() * 100), value(value){
        //Nothing else to init
    }

    ComplexApprox(T real, T imag) : eps( std::numeric_limits<float>::epsilon() * 100), value(real, imag){
        //Nothing else to init
    }

    ComplexApprox(const ComplexApprox& other) : eps(other.eps), value(other.value){
        //Nothing else to init
    }

    friend bool operator==(std::complex<T> lhs, const ComplexApprox& rhs){
        return
                fabs(lhs.real() - rhs.value.real()) < rhs.eps * (1.0 + std::max(fabs(lhs.real()), fabs(rhs.value.real())))
            &&  fabs(lhs.imag() - rhs.value.imag()) < rhs.eps * (1.0 + std::max(fabs(lhs.imag()), fabs(rhs.value.imag())));
    }

    friend bool operator==(const ComplexApprox& lhs, std::complex<T> rhs){
        return operator==(rhs, lhs);
    }

    friend bool operator!=(std::complex<T> lhs, const ComplexApprox& rhs){
        return !operator==(lhs, rhs);
    }

    friend bool operator!=(const ComplexApprox& lhs, std::complex<T> rhs){
        return !operator==(rhs, lhs);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ComplexApprox(" << Catch::toString(value) << ")";
        return oss.str();
    }

private:
    double eps;
    std::complex<T> value;
};

namespace Catch {

template<>
inline std::string toString<ComplexApprox<float>>(const ComplexApprox<float>& value){
    return value.toString();
}

template<>
inline std::string toString<ComplexApprox<double>>(const ComplexApprox<double>& value){
    return value.toString();
}

} // end of namespace Catch
