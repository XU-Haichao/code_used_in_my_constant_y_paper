#pragma once

#include "vec3.hpp"

namespace ic {

struct Photon {
    double energy = 0.0;
    Vec3 direction{0.0, 0.0, 1.0};
};

Photon make_incident_monoenergetic_photon(
    double energy,
    const Vec3& direction = Vec3{0.0, 0.0, 1.0});

}  // namespace ic
