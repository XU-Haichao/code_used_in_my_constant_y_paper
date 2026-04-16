#pragma once

#include <random>
#include <string>

#include "photon.hpp"

namespace ic {

bool is_supported_slab_injection_model(const std::string& model);
Vec3 sample_slab_injection_direction(const std::string& model, std::mt19937_64& rng);
Photon inject_photon_from_lower_boundary(double energy,
                                         const std::string& model,
                                         std::mt19937_64& rng);

}  // namespace ic
