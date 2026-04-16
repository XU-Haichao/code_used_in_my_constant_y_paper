#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ic {

class Histogram {
public:
    Histogram(std::size_t num_bins, double min_value, double max_value);

    void fill(double value, double weight = 1.0);
    void write_csv(const std::string& path) const;

    [[nodiscard]] double in_range_total() const;
    [[nodiscard]] double underflow() const { return underflow_; }
    [[nodiscard]] double overflow() const { return overflow_; }

private:
    [[nodiscard]] std::size_t index_for(double value) const;

    double min_value_ = 0.0;
    double max_value_ = 0.0;
    double bin_width_ = 0.0;
    std::vector<double> counts_;
    double underflow_ = 0.0;
    double overflow_ = 0.0;
};

}  // namespace ic
