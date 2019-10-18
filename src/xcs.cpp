#define __USE_MINGW_ANSI_STDIO 0
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <cstddef>

#include <xxr/xcs.hpp>
#include <xxr/helper/experiment_helper.hpp>
#include <cxxopts.hpp>

using namespace xxr;
using namespace xxr::xcs_impl;

int main(int argc, char *argv[])
{
    Constants constants;

    // Parse command line arguments
    cxxopts::Options options(argv[0], "XCS Classifier System");

    options
        .allow_unrecognised_options()
        .add_options()
        ("summary-interval", "The iteration interval of summary log output", cxxopts::value<uint64_t>()->default_value("5000"), "COUNT")
        ("p,prefix", "The filename prefix for log file output", cxxopts::value<std::string>()->default_value(""), "PREFIX")
        ("S,soutput", "The filename of summary log csv output", cxxopts::value<std::string>()->default_value("summary.csv"), "FILENAME")
        ("o,coutput", "The filename of classifier csv output", cxxopts::value<std::string>()->default_value("classifier.csv"), "FILENAME")
        ("r,routput", "The filename of reward log csv output", cxxopts::value<std::string>()->default_value("reward.csv"), "FILENAME")
        ("E,seoutput", "The filename of system error log csv output", cxxopts::value<std::string>()->default_value("syserr.csv"), "FILENAME")
        ("n,noutput", "The filename of macro-classifier count log csv output", cxxopts::value<std::string>()->default_value("num.csv"), "FILENAME")
        ("nsoutput", "The filename of number-of-step log csv output in the multi-step problem", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("cinput", "The classifier csv filename for initial population", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("resume", "Whether to use initial classifiers (--cinput) to resume previous experiment (\"false\": initialize p/epsilon/F/exp/ts/as to defaults, \"true\": do not initialize values and set system time stamp to the same as that of the latest classifier)", cxxopts::value<bool>()->default_value("false"), "true/false")
        ("m,mux", "Use the multiplexer problem", cxxopts::value<int>(), "LENGTH")
        ("parity", "Use the even-parity problem", cxxopts::value<int>(), "LENGTH")
        ("blc", "Use the block world problem", cxxopts::value<std::string>(), "FILENAME")
        ("blc-3bit", "Use 3-bit representation for each block in a situation", cxxopts::value<bool>()->default_value("false"), "true/false")
        ("blc-diag", "Allow diagonal actions in the block world problem", cxxopts::value<bool>()->default_value("true"), "true/false")
        ("blc-output-best", "Output the result of the desired action for blocks in the block world problem", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("blc-output-best-uni", "Use UTF-8 square & arrow characters for --blc-output", cxxopts::value<bool>()->default_value("false"), "true/false")
        ("blc-output-trace", "Output the coordinate of the animat in the block world problem", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("c,csv", "Use the csv file", cxxopts::value<std::string>(), "FILENAME")
        ("e,csv-eval", "Use the csv file for evaluation", cxxopts::value<std::string>(), "FILENAME")
        ("csv-random", "Whether to choose lines in random order from the csv file", cxxopts::value<bool>()->default_value("true"), "true/false")
        ("max-step", "The maximum number of steps in the multi-step problem", cxxopts::value<uint64_t>()->default_value("50"))
        ("i,iter", "The number of iterations", cxxopts::value<uint64_t>()->default_value("20000"), "COUNT")
        ("condense-iter", "The number of iterations for the Wilson's rule condensation method (chi=0, mu=0) after normal iterations", cxxopts::value<uint64_t>()->default_value("0"), "COUNT")
        ("avg-seeds", "The number of different random seeds for averaging the reward and the macro-classifier count", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("explore", "The number of exploration performed in each iteration", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("exploit", "The number of exploitation (= test mode) performed in each iteration (set \"0\" if you don't need evaluation)", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("exploit-upd", "Whether to update classifier parameters in test mode (\"auto\": false for single-step & true for multi-step)", cxxopts::value<std::string>()->default_value("auto"), "auto/true/false")
        ("sma", "The width of the simple moving average for the reward log", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("a,action", "The available action choices for csv (comma-separated, integer only)", cxxopts::value<std::string>(), "ACTIONS")
        ("N,max-population", "The maximum size of the population", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.n)), "COUNT")
        ("alpha", "The fall of rate in the fitness evaluation", cxxopts::value<double>()->default_value(std::to_string(constants.alpha)), "ALPHA")
        ("beta", "The learning rate for updating fitness, prediction, prediction error, and action set size estimate in XCS's classifiers", cxxopts::value<double>()->default_value(std::to_string(constants.beta)), "BETA")
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
        ("s,p-sharp", "The probability of using a don't care symbol in an allele when covering", cxxopts::value<double>()->default_value(std::to_string(constants.dontCareProbability)), "P_SHARP")
        ("initial-prediction", "The initial prediction value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialPrediction)), "P_I")
        ("initial-prediction-error", "The initial prediction error value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialEpsilon)), "EPSILON_I")
        ("initial-fitness", "The initial fitness value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialFitness)), "F_I")
        ("p-explr", "The probability during action selection of choosing the action uniform randomly", cxxopts::value<double>()->default_value(std::to_string(constants.exploreProbability)), "P_EXPLR")
        ("theta-mna", "The minimal number of actions that must be present in a match set [M], or else covering will occur. Use \"0\" to set automatically.", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.thetaMna)), "THETA_MNA")
        ("do-ga-subsumption", "Whether offspring are to be tested for possible logical subsumption by parents", cxxopts::value<bool>()->default_value(constants.doGASubsumption ? "true" : "false"), "true/false")
        ("do-action-set-subsumption", "Whether action sets are to be tested for subsuming classifiers", cxxopts::value<bool>()->default_value(constants.doActionSetSubsumption ? "true" : "false"), "true/false")
        ("do-action-mutation", "Whether to apply mutation to the action", cxxopts::value<bool>()->default_value(constants.doActionMutation ? "true" : "false"), "true/false")
        ("mam", "Whether to use the moyenne adaptive modifee (MAM) for updating the prediction and the prediction error of classifiers", cxxopts::value<bool>()->default_value(constants.useMAM ? "true" : "false"), "true/false")
        ("h,help", "Show this help");

    auto result = options.parse(argc, argv);

    // Set constants
    if (result.count("max-population"))
        constants.n = result["max-population"].as<uint64_t>();
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
    if (result.count("p-sharp"))
        constants.dontCareProbability = result["p-sharp"].as<double>();
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
    if (result.count("mam"))
        constants.useMAM = result["mam"].as<bool>();

    bool isEnvironmentSpecified = (result.count("mux") || result.count("parity") || result.count("blc") || result.count("csv"));

    // Determine crossover method
    if (result["x-method"].as<std::string>() == "uniform")
    {
        constants.crossoverMethod = XCSConstants::CrossoverMethod::UNIFORM_CROSSOVER;
    }
    else if (result["x-method"].as<std::string>() == "one-point")
    {
        constants.crossoverMethod = XCSConstants::CrossoverMethod::ONE_POINT_CROSSOVER;
    }
    else if (result["x-method"].as<std::string>() == "two-point")
    {
        constants.crossoverMethod = XCSConstants::CrossoverMethod::TWO_POINT_CROSSOVER;
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
        updateInExploitation = result.count("blc");
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
        // Output parameters
        std::cout << "[ XCS General Parameters ]" << std::endl;
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
        std::cout << "             P_# = " << constants.dontCareProbability << std::endl;
        std::cout << "             p_I = " << constants.initialPrediction << std::endl;
        std::cout << "       epsilon_I = " << constants.initialEpsilon << std::endl;
        std::cout << "             F_I = " << constants.initialFitness << std::endl;
        std::cout << "         p_explr = " << constants.exploreProbability << std::endl;
        if (constants.thetaMna != 0) std::cout << "       theta_mna = " << constants.thetaMna << std::endl;
        std::cout << " doGASubsumption = " << (constants.doGASubsumption ? "true" : "false") << std::endl;
        std::cout << " doASSubsumption = " << (constants.doActionSetSubsumption ? "true" : "false") << std::endl;
        std::cout << " crossoverMethod = ";
        switch (constants.crossoverMethod)
        {
        case XCSConstants::CrossoverMethod::UNIFORM_CROSSOVER:
            std::cout << "uniform" << std::endl;
            break;
        case XCSConstants::CrossoverMethod::ONE_POINT_CROSSOVER:
            std::cout << "one-point" << std::endl;
            break;
        case XCSConstants::CrossoverMethod::TWO_POINT_CROSSOVER:
            std::cout << "two-point" << std::endl;
            break;
        }
        std::cout << std::endl;

        // Output optional settings
        std::stringstream ss;
        if (constants.tau > 0.0 && constants.tau <= 1.0)
            ss << "             tau = " << constants.tau << std::endl;
        if (!constants.doActionMutation)
            ss << "doActionMutation = false" << std::endl;
        if (!constants.useMAM)
            ss << "          useMAM = false" << std::endl;
        std::string str = ss.str();
        if (!str.empty())
        {
            std::cout << "[ XCS Optional Settings ]" << std::endl;
            std::cout << str << std::endl;
        }
    }

    ExperimentSettings settings;
    settings.seedCount = result["avg-seeds"].as<uint64_t>();
    settings.explorationCount = result["explore"].as<uint64_t>();
    settings.exploitationCount = result["exploit"].as<uint64_t>();
    settings.updateInExploitation = updateInExploitation;
    settings.summaryInterval = result["summary-interval"].as<uint64_t>();
    settings.outputFilenamePrefix = result["prefix"].as<std::string>();
    settings.outputSummaryToStdout = true;
    settings.outputSummaryFilename = result["soutput"].as<std::string>();
    settings.outputRewardFilename = result["routput"].as<std::string>();
    settings.outputSystemErrorFilename = result["seoutput"].as<std::string>();
    settings.outputPopulationSizeFilename = result["noutput"].as<std::string>();
    settings.outputStepCountFilename = result["nsoutput"].as<std::string>();
    settings.inputClassifierFilename = result["cinput"].as<std::string>();
    settings.useInputClassifierToResume = result["resume"].as<bool>();
    settings.smaWidth = result["sma"].as<uint64_t>();

    std::unique_ptr<AbstractExperimentHelper> experimentHelper;

    // Use multiplexer problem
    if (result.count("mux"))
    {
        std::vector<std::unique_ptr<MultiplexerEnvironment>> explorationEnvironments;
        std::vector<std::unique_ptr<MultiplexerEnvironment>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<MultiplexerEnvironment>(result["mux"].as<int>()));
            exploitationEnvironments.push_back(std::make_unique<MultiplexerEnvironment>(result["mux"].as<int>()));
        }

        experimentHelper = std::make_unique<ExperimentHelper<XCS<bool, bool>, MultiplexerEnvironment>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments)
        );
    }

    // Use even-parity problem
    if (result.count("parity"))
    {
        std::vector<std::unique_ptr<EvenParityEnvironment>> explorationEnvironments;
        std::vector<std::unique_ptr<EvenParityEnvironment>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<EvenParityEnvironment>(result["parity"].as<int>()));
            exploitationEnvironments.push_back(std::make_unique<EvenParityEnvironment>(result["parity"].as<int>()));
        }

        experimentHelper = std::make_unique<ExperimentHelper<XCS<bool, bool>, EvenParityEnvironment>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments)
        );
    }

    // Use block world problem
    std::ofstream blockWorldTraceLogStream;
    if (result.count("blc"))
    {
        std::vector<std::unique_ptr<BlockWorldEnvironment>> explorationEnvironments;
        std::vector<std::unique_ptr<BlockWorldEnvironment>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<BlockWorldEnvironment>(result["blc"].as<std::string>(), result["max-step"].as<uint64_t>(), result["blc-3bit"].as<bool>(), result["blc-diag"].as<bool>()));
            exploitationEnvironments.push_back(std::make_unique<BlockWorldEnvironment>(result["blc"].as<std::string>(), result["max-step"].as<uint64_t>(), result["blc-3bit"].as<bool>(), result["blc-diag"].as<bool>()));
        }

        // Prepare trace output
        bool outputTraceLog = !result["blc-output-trace"].as<std::string>().empty();
        if (outputTraceLog)
        {
            blockWorldTraceLogStream.open(result["blc-output-trace"].as<std::string>());
        }
        std::function<void(BlockWorldEnvironment &)> explorationCallback = [outputTraceLog, &blockWorldTraceLogStream](BlockWorldEnvironment & env) {
            if (outputTraceLog)
            {
                if (env.lastStep() <= 1)
                {
                    blockWorldTraceLogStream << "(" << env.lastInitialX() << "," << env.lastInitialY() << ")";
                }
                blockWorldTraceLogStream << "(" << env.lastX() << "," << env.lastY() << ")";
                if (env.isEndOfProblem())
                {
                    blockWorldTraceLogStream << " Explore" << std::endl;
                }
            }
        };
        std::function<void(BlockWorldEnvironment &)> exploitationCallback = [outputTraceLog, &blockWorldTraceLogStream](BlockWorldEnvironment & env) {
            if (outputTraceLog)
            {
                if (env.lastStep() <= 1)
                {
                    blockWorldTraceLogStream << "(" << env.lastInitialX() << "," << env.lastInitialY() << ")";
                }
                blockWorldTraceLogStream << "(" << env.lastX() << "," << env.lastY() << ")";
                if (env.isEndOfProblem())
                {
                    blockWorldTraceLogStream << " Exploit" << std::endl;
                }
            }
        };

        experimentHelper = std::make_unique<ExperimentHelper<XCS<bool, int>, BlockWorldEnvironment>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments),
            explorationCallback,
            exploitationCallback
        );
    }

    // Use csv file
    if (result.count("csv"))
    {
        // Get available action choices
        if (!result.count("action"))
        {
            std::cout << "Error: Available action list (--action) is not specified." << std::endl;
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
                std::cout << "Error: Action must be an integer." << std::endl;
                exit(1);
            }
        }

        std::string filename = result["csv"].as<std::string>();
        std::string evaluationCsvFilename = filename;
        if (result.count("csv-eval"))
        {
            evaluationCsvFilename = result["csv-eval"].as<std::string>();
        }

        std::vector<std::unique_ptr<DatasetEnvironment<int, int>>> explorationEnvironments;
        std::vector<std::unique_ptr<DatasetEnvironment<int, int>>> exploitationEnvironments;
        for (std::size_t i = 0; i < settings.seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<DatasetEnvironment<int, int>>(CSV::readDataset<int, int>(filename), availableActions, result["csv-random"].as<bool>()));
            exploitationEnvironments.push_back(std::make_unique<DatasetEnvironment<int, int>>(CSV::readDataset<int, int>(evaluationCsvFilename), availableActions, result["csv-random"].as<bool>()));
        }

        experimentHelper = std::make_unique<ExperimentHelper<XCS<int, int>, DatasetEnvironment<int, int>>>(
            settings,
            constants,
            std::move(explorationEnvironments),
            std::move(exploitationEnvironments)
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
        std::string filename = settings.outputFilenamePrefix + result["coutput"].as<std::string>();

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

    // Save block world problem log
    if (result.count("blc"))
    {
        auto & experimentHelperRef = dynamic_cast<ExperimentHelper<XCS<bool, int>, BlockWorldEnvironment> &>(*experimentHelper);
        auto & experiment = experimentHelperRef.experimentAt(0);
        auto & environment = experimentHelperRef.exploitationEnvironmentAt(0);

        if (!result["blc-output-best"].as<std::string>().empty())
        {
            std::ofstream ofs(result["blc-output-best"].as<std::string>());

            bool useUnicode = result["blc-output-best-uni"].as<bool>();

            for (std::size_t y = 0; y < environment.worldHeight(); ++y)
            {
                for (std::size_t x = 0; x < environment.worldWidth(); ++x)
                {
                    if (environment.isEmpty(x, y))
                    {
                        // Output the selected action
                        auto situation = environment.situation(x, y);
                        int action = experiment.exploit(situation);
                        if (useUnicode)
                        {
                            switch (action)
                            {
                            case 0:
                                ofs << u8"↑";
                                break;

                            case 1:
                                ofs << u8"↗";
                                break;

                            case 2:
                                ofs << u8"→";
                                break;

                            case 3:
                                ofs << u8"↘";
                                break;

                            case 4:
                                ofs << u8"↓";
                                break;

                            case 5:
                                ofs << u8"↙";
                                break;

                            case 6:
                                ofs << u8"←";
                                break;

                            case 7:
                                ofs << u8"↖";
                                break;

                            default:
                                ofs << u8"？";
                            }
                        }
                        else
                        {
                            ofs << action;
                        }
                    }
                    else
                    {
                        // Obstacle or Food
                        unsigned char c = environment.getBlock(x, y);
                        if (useUnicode)
                        {
                            switch (c)
                            {
                            case 'Q':
                                ofs << u8"◆";
                                break;

                            case 'F':
                                ofs << u8"Ｆ";
                                break;

                            case 'G':
                                ofs << u8"Ｇ";
                                break;

                            default:
                                ofs << u8"■";
                            }
                        }
                        else
                        {
                            ofs << c;
                        }
                    }
                }
                ofs << std::endl;
            }
        }
    }

    return 0;
}
