#include "histogram.hpp"

#include <fstream>
#include <numeric>
#include <stdexcept>

namespace ic {

Histogram::Histogram(std::size_t num_bins, double min_value, double max_value)
    : min_value_(min_value), max_value_(max_value), counts_(num_bins, 0.0) {
    if (num_bins == 0) {
        throw std::runtime_error("histogram requires at least one bin");
    }
    if (!(max_value > min_value)) {
        throw std::runtime_error("histogram max must be larger than min");
    }
    bin_width_ = (max_value_ - min_value_) / static_cast<double>(counts_.size());
}

void Histogram::fill(double value, double weight) {
    if (value < min_value_) {
        underflow_ += weight;
        return;
    }
    if (value >= max_value_) {
        overflow_ += weight;
        return;
    }
    counts_.at(index_for(value)) += weight;
}

void Histogram::write_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open histogram output: " + path);
    }

    out << "bin_index,low_edge,high_edge,center,count\n";
    for (std::size_t i = 0; i < counts_.size(); ++i) {
        const double low = min_value_ + static_cast<double>(i) * bin_width_;
        const double high = low + bin_width_;
        const double center = 0.5 * (low + high);
        out << i << ',' << low << ',' << high << ',' << center << ',' << counts_[i] << "\n";
    }
    out << "underflow,,,," << underflow_ << "\n";
    out << "overflow,,,," << overflow_ << "\n";
}

double Histogram::in_range_total() const {
    return std::accumulate(counts_.begin(), counts_.end(), 0.0);
}

std::size_t Histogram::index_for(double value) const {
    const double offset = (value - min_value_) / bin_width_;
    std::size_t index = static_cast<std::size_t>(offset);
    if (index >= counts_.size()) {
        index = counts_.size() - 1;
    }
    return index;
}

}  // namespace ic
