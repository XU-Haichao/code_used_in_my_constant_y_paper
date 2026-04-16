#include "photon.hpp"

namespace ic {

Photon make_incident_monoenergetic_photon(double energy, const Vec3& direction) {
    Photon photon;
    photon.energy = energy;
    photon.direction = direction.normalized();
    return photon;
}

}  // namespace ic
