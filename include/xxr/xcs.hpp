#pragma once

#include "xcs/experiment.hpp"

#include "environment/multiplexer_environment.hpp"
#include "environment/even_parity_environment.hpp"
#include "environment/majority_on_environment.hpp"
#include "environment/block_world_environment.hpp"
#include "environment/dataset_environment.hpp"
#include "environment/csv_environment.hpp"

namespace xxr
{

    template <typename T = int, typename Action = int>
    using XCS = xcs_impl::Experiment<T, Action>;

    using XCSConstants = xcs_impl::Constants;

}
