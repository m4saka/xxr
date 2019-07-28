#pragma once
#include <cstddef>

struct ExperimentSettings
{
    // The number of different random seeds for averaging the reward and the macro-classifier count
    std::size_t seedCount = 1;

    // The number of exploration performed in each iteration
    std::size_t explorationCount = 1;

    // The number of exploitation (= test mode) performed in each iteration (set "0" if you don't need evaluation)
    std::size_t exploitationCount = 1;

    // Whether to update classifier parameters in test mode (usually "false" for single-step problems and "true" for multi-step problems)
    bool updateInExploitation = false;

    // The iteration interval of average log output
    std::size_t summaryInterval = 5000;

    // The prefix of filename
    std::string outputFilenamePrefix;

    // The filename of reward log csv output
    std::string outputRewardFilename = "reward.csv";

    // The filename of macro-classifier count log csv output
    std::string outputPopulationSizeFilename = "num.csv";

    // The filename of system error log csv output
    std::string outputSystemErrorFilename = "syserr.csv";

    // The filename of number-of-step log csv output in the multi-step problem
    std::string outputStepCountFilename;

    // The classifier csv filename for initial population
    std::string inputClassifierFilename;

    // Whether to use initial classifiers (--cinput) to resume previous experiment
    //   "false": initialize p/epsilon/F/exp/ts/as to defaults
    //   "true": do not initialize values and set system time stamp to the same as that of the latest classifier
    bool useInputClassifierToResume = true;

    // The width of the simple moving average for the reward log
    std::size_t smaWidth = 1;
};
