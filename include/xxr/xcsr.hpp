#pragma once

#include "xcsr/repr.hpp"
#include "xcsr/experiment.hpp"
#include "environment/real_multiplexer_environment.hpp"
#include "environment/checkerboard_environment.hpp"
#include "environment/rotated_checkerboard_environment.hpp"
#include "environment/dataset_environment.hpp"
#include "environment/csv_environment.hpp"

namespace xxr
{

    constexpr auto CSR = xcsr_impl::Repr::CSR;
    constexpr auto OBR = xcsr_impl::Repr::OBR;
    constexpr auto UBR = xcsr_impl::Repr::UBR;

    // XCSR Classifier System
    template <typename T = double, typename Action = int>
    using XCSR = xcsr_impl::Experiment<T, Action>;

    using XCSRRepr = xcsr_impl::Repr;
    using XCSRConstants = xcsr_impl::Constants;

}
