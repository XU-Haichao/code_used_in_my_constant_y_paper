#include <cstdlib>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <XSFunctions/MZCompRefl.h>
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <XSFunctions/Utilities/xsFortran.h>

namespace {

void initialize_xspec_runtime() {
    static bool initialized = false;
    if (initialized) {
        return;
    }

    const char* headas_env = std::getenv("HEADAS");
    if (!headas_env || !headas_env[0]) {
        throw std::runtime_error("HEADAS is not set for the reflect bridge.");
    }

    FNINIT();
    FPCHAT(0);

    int ierr = 0;
    FPSOLR("angr", &ierr);
    if (ierr != 0) {
        throw std::runtime_error("FPSOLR failed while initializing the reflect bridge.");
    }
    FPXSCT("bcmc", &ierr);
    if (ierr != 0) {
        throw std::runtime_error("FPXSCT failed while initializing the reflect bridge.");
    }

    initialized = true;
}

}  // namespace

extern "C" int mz_reflect_spectrum(
    int n,
    const double* x_in,
    const double* spinc_in,
    double cos_incl,
    double abund,
    double fe_abund,
    double x_max,
    double* spref_out
) {
    try {
        initialize_xspec_runtime();

        if (n <= 0 || !x_in || !spinc_in || !spref_out) {
            return 1;
        }

        RealArray x(static_cast<std::size_t>(n));
        RealArray spinc(static_cast<std::size_t>(n));
        RealArray sptot(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] = x_in[i];
            spinc[static_cast<std::size_t>(i)] = spinc_in[i];
        }

        calcCompReflTotalFlux(
            std::string("reflect"),
            -1.0,
            cos_incl,
            abund,
            fe_abund,
            0.0,
            0.0,
            x_max,
            x,
            spinc,
            sptot
        );

        for (int i = 0; i < n; ++i) {
            spref_out[static_cast<std::size_t>(i)] = sptot[static_cast<std::size_t>(i)];
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "mz_reflect_spectrum exception: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "mz_reflect_spectrum exception: unknown" << std::endl;
        return 2;
    }
}
