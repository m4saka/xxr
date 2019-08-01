#define __USE_MINGW_ANSI_STDIO 0
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <cstddef>

#include <xxr/xcsr.hpp>
#include <xxr/helper/experiment_helper.hpp>
#include <cxxopts.hpp>

using namespace xxr;
using namespace xxr::xcsr_impl;

int main(int argc, char *argv[])
{
    xcsr_impl::Constants constants;

    // Parse command line arguments
    cxxopts::Options options(argv[0], "XCSR Classifier System");

    options
        .allow_unrecognised_options()
        .add_options()
        ("summary-interval", "The iteration interval of summary log output", cxxopts::value<uint64_t>()->default_value("5000"), "COUNT")
        ("S,soutput", "The filename of summary log csv output", cxxopts::value<std::string>()->default_value("summary.csv"), "FILENAME")
        ("o,coutput", "The filename of classifier csv output", cxxopts::value<std::string>()->default_value("classifier.csv"), "FILENAME")
        ("r,routput", "The filename of reward log csv output", cxxopts::value<std::string>()->default_value("reward.csv"), "FILENAME")
        ("E,seoutput", "The filename of system error log csv output", cxxopts::value<std::string>()->default_value("syserr.csv"), "FILENAME")
        ("n,noutput", "The filename of macro-classifier count log csv output", cxxopts::value<std::string>()->default_value("num.csv"), "FILENAME")
        //("nsoutput", "The filename of number-of-step log csv output in the multi-step problem", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("cinput", "The classifier csv filename for initial population", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("resume", "Whether to use initial classifiers (--cinput) to resume previous experiment (\"false\": initialize p/epsilon/F/exp/ts/as to defaults, \"true\": do not initialize values and set system time stamp to the same as that of the latest classifier)", cxxopts::value<bool>()->default_value("false"), "true/false")
        ("m,mux", "Use the real multiplexer problem", cxxopts::value<int>(), "LENGTH")
        ("chk", "Use the n-dimentional checkerboard problem", cxxopts::value<int>(), "N")
        ("chk-div", "The division in the checkerboard problem", cxxopts::value<int>(), "DIVISION")
        ("c,csv", "Use the csv file", cxxopts::value<std::string>(), "FILENAME")
        ("e,csv-eval", "Use the csv file for evaluation", cxxopts::value<std::string>(), "FILENAME")
        ("csv-random", "Whether to choose lines in random order from the csv file", cxxopts::value<bool>()->default_value("true"), "true/false")
        ("i,iter", "The number of iterations", cxxopts::value<uint64_t>()->default_value("20000"), "COUNT")
        ("condense-iter", "The number of iterations for the Wilson's rule condensation method (chi=0, mu=0) after normal iterations", cxxopts::value<uint64_t>()->default_value("0"), "COUNT")
        ("avg-seeds", "The number of different random seeds for averaging the reward and the macro-classifier count", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("explore", "The number of exploration performed in each iteration", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("exploit", "The number of exploitation (= test mode) performed in each iteration (set \"0\" if you don't need evaluation)", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("exploit-upd", "Whether to update classifier parameters in test mode (\"auto\": false for single-step & true for multi-step)", cxxopts::value<std::string>()->default_value("auto"), "auto/true/false")
        ("sma", "The width of the simple moving average for the reward log", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("a,action", "The available action choices for csv (comma-separated, integer only)", cxxopts::value<std::string>(), "ACTIONS")
        ("repr", "The XCSR classifier representation (Center-Spread: \"cs\" / Lower-Upper [Ordered Bound]: \"lu\" / Unordered Bound: \"ub\")", cxxopts::value<std::string>()->default_value("cs"), "cs/lu/ub")
        ("N,max-population", "The maximum size of the population", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.n)), "COUNT")
        ("min-value", "The minimum value of each symbol in a situation", cxxopts::value<double>()->default_value(std::to_string(constants.minValue)), "VALUE")
        ("max-value", "The maximum value of each symbol in a situation", cxxopts::value<double>()->default_value(std::to_string(constants.maxValue)), "VALUE")
        ("alpha", "The fall of rate in the fitness evaluation", cxxopts::value<double>()->default_value(std::to_string(constants.alpha)), "ALPHA")
        ("beta", "The learning rate for updating fitness, prediction, prediction error, and action set size estimate in xcsr's classifiers", cxxopts::value<double>()->default_value(std::to_string(constants.beta)), "BETA")
        ("epsilon-0", "The error threshold under which the accuracy of a classifier is set to one", cxxopts::value<double>()->default_value(std::to_string(constants.alpha)), "EPSILON_0")
        ("nu", "The exponent in the power function for the fitness evaluation", cxxopts::value<double>()->default_value(std::to_string(constants.nu)), "NU")
        ("gamma", "The discount rate in multi-step problems", cxxopts::value<double>()->default_value(std::to_string(constants.gamma)), "GAMMA")
        ("theta-ga", "The threshold for the GA application in an action set", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.thetaGA)), "THETA_GA")
        ("chi", "The probability of applying crossover", cxxopts::value<double>()->default_value(std::to_string(constants.chi)), "CHI")
        ("x-method", "The method of crossover operator", cxxopts::value<std::string>()->default_value("two-point"), "uniform/one-point/two-point")
        ("mu", "The probability of mutating one allele and the action", cxxopts::value<double>()->default_value(std::to_string(constants.mu)), "MU")
        ("theta-del", "The experience threshold over which the fitness of a classifier may be considered in its deletion probability", cxxopts::value<double>()->default_value(std::to_string(constants.thetaDel)), "THETA_DEL")
        ("delta", "The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion", cxxopts::value<double>()->default_value(std::to_string(constants.delta)), "DELTA")
        ("theta-sub", "The experience of a classifier required to be a subsumer", cxxopts::value<double>()->default_value(std::to_string(constants.thetaSub)), "THETA_SUB")
        ("tau", "The tournament size for selection [Butz et al., 2003] (set \"0\" to use the roulette-wheel selection)", cxxopts::value<double>()->default_value(std::to_string(constants.tau)), "TAU")
        ("s-0", "The maximum value of a spread in the covering operator", cxxopts::value<double>()->default_value(std::to_string(constants.coveringMaxSpread)), "S_0")
        ("max-mutation", "The maximum change of a spread value or a center value in mutation", cxxopts::value<double>()->default_value(std::to_string(constants.mutationMaxChange)), "M")
        ("tol-sub", "The tolerance range of the upper and lower bound in the subsumption", cxxopts::value<double>()->default_value(std::to_string(constants.subsumptionTolerance)), "TOL_SUB")
        ("initial-prediction", "The initial prediction value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialPrediction)), "P_I")
        ("initial-prediction-error", "The initial prediction error value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialEpsilon)), "EPSILON_I")
        ("initial-fitness", "The initial fitness value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialFitness)), "F_I")
        ("p-explr", "The probability during action selection of choosing the action uniform randomly", cxxopts::value<double>()->default_value(std::to_string(constants.exploreProbability)), "P_EXPLR")
        ("theta-mna", "The minimal number of actions that must be present in a match set [M], or else covering will occur. Use \"0\" to set automatically.", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.thetaMna)), "THETA_MNA")
        ("do-ga-subsumption", "Whether offspring are to be tested for possible logical subsumption by parents", cxxopts::value<bool>()->default_value(constants.doGASubsumption ? "true" : "false"), "true/false")
        ("do-action-set-subsumption", "Whether action sets are to be tested for subsuming classifiers", cxxopts::value<bool>()->default_value(constants.doActionSetSubsumption ? "true" : "false"), "true/false")
        ("do-action-mutation", "Whether to apply mutation to the action", cxxopts::value<bool>()->default_value(constants.doActionMutation ? "true" : "false"), "true/false")
        ("do-range-restriction", "Whether to restrict the range of the condition to the interval [min-value, max-value) in the covering and mutation operator (ignored when --repr=csr)", cxxopts::value<bool>()->default_value(constants.doRangeRestriction ? "true" : "false"), "true/false")
        ("do-covering-random-range-truncation", "Whether to truncate the covering random range before generating random intervals if the interval [x-s_0, x+s_0) is not contained in [min-value, max-value).  \"false\" is common for this option, but the covering operator can generate too many maximum-range intervals if s_0 is larger than (max-value - min-value) / 2.  Choose \"true\" to avoid the random bias in this situation.  (ignored when --repr=csr)", cxxopts::value<bool>()->default_value(constants.doCoveringRandomRangeTruncation ? "true" : "false"), "true/false")
        ("mam", "Whether to use the moyenne adaptive modifee (MAM) for updating the prediction and the prediction error of classifiers", cxxopts::value<bool>()->default_value(constants.useMAM ? "true" : "false"), "true/false")
        ("h,help", "Show this help");

    auto result = options.parse(argc, argv);

    // Set constants
    if (result.count("max-population"))
        constants.n = result["max-population"].as<uint64_t>();
    if (result.count("min-value"))
        constants.minValue = result["min-value"].as<double>();
    if (result.count("max-value"))
        constants.maxValue = result["max-value"].as<double>();
    if (result.count("alpha"))
        constants.alpha = result["alpha"].as<double>();
    if (result.count("beta"))
        constants.beta = result["beta"].as<double>();
    if (result.count("epsilon-0"))
        constants.epsilonZero = result["epsilon-0"].as<double>();
    if (result.count("nu"))
        constants.nu = result["nu"].as<double>();
    if (result.count("gamma"))
        constants.gamma = result["gamma"].as<double>();
    if (result.count("theta-ga"))
        constants.thetaGA = result["theta-ga"].as<uint64_t>();
    if (result.count("chi"))
        constants.chi = result["chi"].as<double>();
    if (result.count("mu"))
        constants.mu = result["mu"].as<double>();
    if (result.count("theta-del"))
        constants.thetaDel = result["theta-del"].as<double>();
    if (result.count("delta"))
        constants.delta = result["delta"].as<double>();
    if (result.count("theta-sub"))
        constants.thetaSub = result["theta-sub"].as<double>();
    if (result.count("tau"))
        constants.tau = result["tau"].as<double>();
    if (result.count("s-0"))
        constants.coveringMaxSpread = result["s-0"].as<double>();
    if (result.count("max-mutation"))
        constants.mutationMaxChange = result["max-mutation"].as<double>();
    if (result.count("tol-sub"))
        constants.subsumptionTolerance = result["tol-sub"].as<double>();
    if (result.count("initial-prediction"))
        constants.initialPrediction = result["initial-prediction"].as<double>();
    if (result.count("initial-prediction-error"))
        constants.initialEpsilon = result["initial-prediction-error"].as<double>();
    if (result.count("initial-fitness"))
        constants.initialFitness = result["initial-fitness"].as<double>();
    if (result.count("p-explr"))
        constants.exploreProbability = result["p-explr"].as<double>();
    if (result.count("theta-mna"))
        constants.thetaMna = result["theta-mna"].as<uint64_t>();
    if (result.count("do-ga-subsumption"))
        constants.doGASubsumption = result["do-ga-subsumption"].as<bool>();
    if (result.count("do-action-set-subsumption"))
        constants.doActionSetSubsumption = result["do-action-set-subsumption"].as<bool>();
    if (result.count("do-action-mutation"))
        constants.doActionMutation = result["do-action-mutation"].as<bool>();
    if (result.count("do-range-restriction"))
        constants.doRangeRestriction = result["do-range-restriction"].as<bool>();
    if (result.count("do-covering-random-range-truncation"))
        constants.doCoveringRandomRangeTruncation = result["do-covering-random-range-truncation"].as<bool>();
    if (result.count("mam"))
        constants.useMAM = result["mam"].as<bool>();

    bool isEnvironmentSpecified = (result.count("mux") || result.count("csv") || result.count("chk"));

    // Determine crossover method
    if (result["x-method"].as<std::string>() == "uniform")
    {
        constants.crossoverMethod = XCSRConstants::CrossoverMethod::UNIFORM_CROSSOVER;
    }
    else if (result["x-method"].as<std::string>() == "one-point")
    {
        constants.crossoverMethod = XCSRConstants::CrossoverMethod::ONE_POINT_CROSSOVER;
    }
    else if (result["x-method"].as<std::string>() == "two-point")
    {
        constants.crossoverMethod = XCSRConstants::CrossoverMethod::TWO_POINT_CROSSOVER;
    }
    else
    {
        std::cerr << "Error: Unknown value for --x-method (" << result["x-method"].as<std::string>() << ")" << std::endl;
        exit(1);
    }

    // Determine whether to update classifier parameters in exploitation
    bool updateInExploitation;
    if (result["exploit-upd"].as<std::string>() == "true")
    {
        updateInExploitation = true;
    }
    else if (result["exploit-upd"].as<std::string>() == "false")
    {
        updateInExploitation = false;
    }
    else if (result["exploit-upd"].as<std::string>() == "auto")
    {
        updateInExploitation = false /* No multi-step problems for now */;
    }
    else
    {
        std::cerr << "Error: Unknown value for --exploit-upd (" << result["exploit-upd"].as<std::string>() << ")" << std::endl;
        exit(1);
    }

    // Show help
    if (result.count("help") || !isEnvironmentSpecified)
    {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }
    else
    {
        // Output general parameters
        std::cout << "[ XCSR General Parameters ]" << std::endl;
        std::cout << "               N = " << constants.n << std::endl;
        std::cout << "            beta = " << constants.beta << std::endl;
        std::cout << "           alpha = " << constants.alpha << std::endl;
        std::cout << "       epsilon_0 = " << constants.epsilonZero << std::endl;
        std::cout << "              nu = " << constants.nu << std::endl;
        std::cout << "           alpha = " << constants.alpha << std::endl;
        std::cout << "           gamma = " << constants.gamma << std::endl;
        std::cout << "        theta_GA = " << constants.thetaGA << std::endl;
        std::cout << "             chi = " << constants.chi << std::endl;
        std::cout << "              mu = " << constants.mu << std::endl;
        std::cout << "       theta_del = " << constants.thetaDel << std::endl;
        std::cout << "           delta = " << constants.delta << std::endl;
        std::cout << "       theta_sub = " << constants.thetaSub << std::endl;
        std::cout << "             s_0 = " << constants.coveringMaxSpread << std::endl;
        std::cout << "               m = " << constants.mutationMaxChange << std::endl;
        std::cout << "             p_I = " << constants.initialPrediction << std::endl;
        std::cout << "       epsilon_I = " << constants.initialEpsilon << std::endl;
        std::cout << "             F_I = " << constants.initialFitness << std::endl;
        std::cout << "         p_explr = " << constants.exploreProbability << std::endl;
        if (constants.thetaMna != 0)
            std::cout << "       theta_mna = " << constants.thetaMna << std::endl;
        std::cout << " doGASubsumption = " << (constants.doGASubsumption ? "true" : "false") << std::endl;
        std::cout << " doASSubsumption = " << (constants.doActionSetSubsumption ? "true" : "false") << std::endl;
        std::cout << " crossoverMethod = ";
        switch (constants.crossoverMethod)
        {
        case XCSRConstants::CrossoverMethod::UNIFORM_CROSSOVER:
            std::cout << "uniform" << std::endl;
            break;
        case XCSRConstants::CrossoverMethod::ONE_POINT_CROSSOVER:
            std::cout << "one-point" << std::endl;
            break;
        case XCSRConstants::CrossoverMethod::TWO_POINT_CROSSOVER:
            std::cout << "two-point" << std::endl;
            break;
        }
        std::cout << std::endl;

        // Output optional settings
        std::stringstream ss;
        if (constants.tau > 0.0 && constants.tau <= 1.0)
            ss << "               tau = " << constants.tau << std::endl;
        if (constants.subsumptionTolerance != 0)
            ss << "           Tol_sub = " << constants.subsumptionTolerance << std::endl;
        if (!constants.doActionMutation)
            ss << "  doActionMutation = false" << std::endl;
        if (!constants.doRangeRestriction)
            ss << "doRangeRestriction = false" << std::endl;
        if (constants.doCoveringRandomRangeTruncation)
            ss << "doCoveringRangeTruncation = true" << std::endl;
        if (!constants.useMAM)
            ss << "            useMAM = false" << std::endl;
        std::string str = ss.str();
        if (!str.empty())
        {
            std::cout << "[ XCSR Optional Settings ]" << std::endl;
            std::cout << str << std::endl;
        }
    }

    ExperimentSettings settings;
    settings.seedCount = result["avg-seeds"].as<uint64_t>();
    settings.explorationCount = result["explore"].as<uint64_t>();
    settings.exploitationCount = result["exploit"].as<uint64_t>();
    settings.updateInExploitation = updateInExploitation;
    settings.summaryInterval = result["summary-interval"].as<uint64_t>();
    settings.outputSummaryToStdout = true;
    settings.outputSummaryFilename = result["soutput"].as<std::string>();
    settings.outputRewardFilename = result["routput"].as<std::string>();
    settings.outputSystemErrorFilename = result["seoutput"].as<std::string>();
    settings.outputPopulationSizeFilename = result["noutput"].as<std::string>();
    //settings.outputStepCountFilename = result["nsoutput"].as<std::string>();
    settings.inputClassifierFilename = result["cinput"].as<std::string>();
    settings.useInputClassifierToResume = result["resume"].as<bool>();
    settings.smaWidth = result["sma"].as<uint64_t>();

    xxr::XCSRRepr repr;
    {
        std::string reprStr = result["repr"].as<std::string>();
        if (reprStr == "csr" || reprStr == "cs")
        {
            repr = xxr::CSR;
        }
        else if (reprStr == "obr" || reprStr == "lu")
        {
            repr = xxr::OBR;
        }
        else if (reprStr == "ubr" || reprStr == "ub")
        {
            repr = xxr::UBR;
        }
        else
        {
            std::cerr << "Error: Unknown representation (" << reprStr << ")" << std::endl;
            exit(1);
        }
    }

    std::unique_ptr<AbstractExperimentHelper> experimentHelper;

    // Use real multiplexer problem
    if (result.count("mux"))
    {
        std::vector<std::unique_ptr<RealMultiplexerEnvironment>> explorationEnvironments;
        std::vector<std::unique_ptr<RealMultiplexerEnvironment>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<RealMultiplexerEnvironment>(result["mux"].as<int>(), true));
            exploitationEnvironments.push_back(std::make_unique<RealMultiplexerEnvironment>(result["mux"].as<int>(), true));
        }

        experimentHelper = std::make_unique<ExperimentHelper<XCSR<double, bool>, RealMultiplexerEnvironment>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments),
            [](RealMultiplexerEnvironment &){},
            [](RealMultiplexerEnvironment &){},
            repr
        );
    }

    // Use checkerboard problem
    if (result.count("chk"))
    {
        if (!result.count("chk-div"))
        {
            std::cerr << "Error: The division in the checkerboard problem (--chk-div) is not specified." << std::endl;
            exit(1);
        }

        std::vector<std::unique_ptr<CheckerboardEnvironment>> explorationEnvironments;
        std::vector<std::unique_ptr<CheckerboardEnvironment>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<CheckerboardEnvironment>(result["chk"].as<int>(), result["chk-div"].as<int>()));
            exploitationEnvironments.push_back(std::make_unique<CheckerboardEnvironment>(result["chk"].as<int>(), result["chk-div"].as<int>()));
        }

        experimentHelper = std::make_unique<ExperimentHelper<XCSR<double, bool>, CheckerboardEnvironment>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments),
            [](CheckerboardEnvironment &){},
            [](CheckerboardEnvironment &){},
            repr
        );
    }

    // Use csv file
    if (result.count("csv"))
    {
        // Get available action choices
        if (!result.count("action"))
        {
            std::cerr << "Error: Available action list (--action) is not specified." << std::endl;
            exit(1);
        }
        std::string availableActionsStr = result["action"].as<std::string>();
        std::string availableActionStr;
        std::stringstream ss(availableActionsStr);
        std::unordered_set<int> availableActions;
        while (std::getline(ss, availableActionStr, ','))
        {
            try
            {
                availableActions.insert(std::stoi(availableActionStr));
            }
            catch (std::exception & e)
            {
                std::cerr << "Error: Action must be an integer." << std::endl;
                exit(1);
            }
        }

        std::string filename = result["csv"].as<std::string>();
        std::string evaluationCsvFilename = filename;
        if (result.count("csv-eval"))
        {
            evaluationCsvFilename = result["csv-eval"].as<std::string>();
        }

        std::vector<std::unique_ptr<DatasetEnvironment<double, int>>> explorationEnvironments;
        std::vector<std::unique_ptr<DatasetEnvironment<double, int>>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<DatasetEnvironment<double, int>>(CSV::readDataset<double, int>(filename), availableActions, result["csv-random"].as<bool>()));
            exploitationEnvironments.push_back(std::make_unique<DatasetEnvironment<double, int>>(CSV::readDataset<double, int>(evaluationCsvFilename), availableActions, result["csv-random"].as<bool>()));
        }

        experimentHelper = std::make_unique<ExperimentHelper<XCSR<double, int>, DatasetEnvironment<double, int>>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments),
            [](DatasetEnvironment<double, int> &){},
            [](DatasetEnvironment<double, int> &){},
            repr
        );
    }

    // Run experiment
    if (experimentHelper)
    {
        uint64_t iterationCount = result["iter"].as<uint64_t>();
        uint64_t condensationIterationCount = result["condense-iter"].as<uint64_t>();

        experimentHelper->runIteration(iterationCount);
        experimentHelper->switchToCondensationMode();
        experimentHelper->runIteration(condensationIterationCount);
    }
    else
    {
        // No target environment (show help)
        std::cout << options.help({"", "Group"}) << std::endl;
        return 1;
    }

    // Save population
    {
        std::string filename = result["coutput"].as<std::string>();

        std::ofstream ofs;
        std::ostream & os = filename.empty() ? std::cout : ofs;
        if (!filename.empty())
        {
            ofs.open(filename);
        }
        if (os)
        {
            experimentHelper->dumpPopulation(0, os);
        }
    }

    return 0;
}
