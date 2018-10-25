#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <cstddef>

#include <cxxopts.hpp>

#include "common.h"
#include "XCS/experiment.h"
#include "environment/multiplexer_environment.h"
#include "environment/csv_environment.h"

using namespace XCS;

int main(int argc, char *argv[])
{
    Constants constants;

    // Parse command line arguments
    cxxopts::Options options(argv[0], "XCS Classifier System");

    options
        .allow_unrecognised_options()
        .add_options()
        ("o,coutput", "Output the classifier csv filename", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("r,routput", "Output the reward log csv filename", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("n,noutput", "Output the macro-classifier count log csv filename", cxxopts::value<std::string>()->default_value(""), "FILENAME")
        ("m,mux", "Use the multiplexer problem", cxxopts::value<int>(), "LENGTH")
        ("c,csv", "Use the csv file", cxxopts::value<std::string>(), "FILENAME")
        ("e,csv-eval", "Use the csv file for evaluation", cxxopts::value<std::string>(), "FILENAME")
        ("csv-random", "Whether to choose lines in random order from the csv file", cxxopts::value<bool>()->default_value("true"), "true/false")
        ("i,iteration", "The number of iterations", cxxopts::value<uint64_t>()->default_value("20000"), "COUNT")
        ("avg-seeds", "The number of different random seeds for averaging the reward and the macro-classifier count", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("explore", "The exploration count for each iteration", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("exploit", "The exploitation count for each iteration (set \"0\" if you don't need evaluation)", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("sma", "The width of the simple moving average for the reward log", cxxopts::value<uint64_t>()->default_value("1"), "COUNT")
        ("a,action", "The available action choices for csv (comma-separated, integer only)", cxxopts::value<std::string>(), "ACTIONS")
        ("N,max-population", "The maximum size of the population", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.maxPopulationClassifierCount)), "COUNT")
        ("alpha", "The fall of rate in the fitness evaluation", cxxopts::value<double>()->default_value(std::to_string(constants.alpha)), "ALPHA")
        ("beta", "The learning rate for updating fitness, prediction, prediction error, and action set size estimate in XCS's classifiers", cxxopts::value<double>()->default_value(std::to_string(constants.learningRate)), "BETA")
        ("epsilon-0", "The error threshold under which the accuracy of a classifier is set to one", cxxopts::value<double>()->default_value(std::to_string(constants.alpha)), "EPSILON_0")
        ("nu", "The exponent in the power function for the fitness evaluation", cxxopts::value<double>()->default_value(std::to_string(constants.nu)), "NU")
        ("gamma", "The discount rate in multi-step problems", cxxopts::value<double>()->default_value(std::to_string(constants.gamma)), "GAMMA")
        ("theta-ga", "The threshold for the GA application in an action set", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.thetaGA)), "THETA_GA")
        ("chi", "The probability of applying crossover", cxxopts::value<double>()->default_value(std::to_string(constants.crossoverProbability)), "CHI")
        ("mu", "The probability of mutating one allele and the action", cxxopts::value<double>()->default_value(std::to_string(constants.mutationProbability)), "MU")
        ("theta-del", "The experience threshold over which the fitness of a classifier may be considered in its deletion probability", cxxopts::value<double>()->default_value(std::to_string(constants.thetaDel)), "THETA_DEL")
        ("delta", "The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion", cxxopts::value<double>()->default_value(std::to_string(constants.delta)), "DELTA")
        ("theta-sub", "The experience of a classifier required to be a subsumer", cxxopts::value<double>()->default_value(std::to_string(constants.thetaSub)), "THETA_SUB")
        ("s,p-sharp", "The probability of using a don't care symbol in an allele when covering", cxxopts::value<double>()->default_value(std::to_string(constants.generalizeProbability)), "P_SHARP")
        ("initial-prediction", "The initial prediction value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialPrediction)), "P_I")
        ("initial-prediction-error", "The initial prediction error value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialPredictionError)), "EPSILON_I")
        ("initial-fitness", "The initial fitness value when generating a new classifier", cxxopts::value<double>()->default_value(std::to_string(constants.initialFitness)), "F_I")
        ("p-explr", "The probability during action selection of choosing the action uniform randomly", cxxopts::value<double>()->default_value(std::to_string(constants.exploreProbability)), "P_EXPLR")
        ("theta-mna", "The minimal number of actions that must be present in a match set [M], or else covering will occur. Use \"0\" to set automatically.", cxxopts::value<uint64_t>()->default_value(std::to_string(constants.thetaMna)), "THETA_MNA")
        ("do-ga-subsumption", "Whether offspring are to be tested for possible logical subsumption by parents", cxxopts::value<bool>()->default_value(constants.doGASubsumption ? "true" : "false"), "true/false")
        ("do-action-set-subsumption", "Whether action sets are to be tested for subsuming classifiers", cxxopts::value<bool>()->default_value(constants.doActionSetSubsumption ? "true" : "false"), "true/false")
        ("do-action-mutation", "Whether to apply mutation to the action", cxxopts::value<bool>()->default_value(constants.doActionMutation ? "true" : "false"), "true/false")
        ("h,help", "Show this help");

    auto result = options.parse(argc, argv);

    // Set constants
    if (result.count("max-population"))
        constants.maxPopulationClassifierCount = result["max-population"].as<uint64_t>();
    if (result.count("alpha"))
        constants.alpha = result["alpha"].as<double>();
    if (result.count("beta"))
        constants.learningRate = result["beta"].as<double>();
    if (result.count("epsilon-0"))
        constants.predictionErrorThreshold = result["epsilon-0"].as<double>();
    if (result.count("nu"))
        constants.nu = result["nu"].as<double>();
    if (result.count("gamma"))
        constants.gamma = result["gamma"].as<double>();
    if (result.count("theta-ga"))
        constants.thetaGA = result["theta-ga"].as<uint64_t>();
    if (result.count("chi"))
        constants.crossoverProbability = result["chi"].as<double>();
    if (result.count("mu"))
        constants.mutationProbability = result["mu"].as<double>();
    if (result.count("theta-del"))
        constants.thetaDel = result["theta-del"].as<double>();
    if (result.count("delta"))
        constants.delta = result["delta"].as<double>();
    if (result.count("theta-sub"))
        constants.thetaSub = result["theta-sub"].as<double>();
    if (result.count("p-sharp"))
        constants.generalizeProbability = result["p-sharp"].as<double>();
    if (result.count("initial-prediction"))
        constants.initialPrediction = result["initial-prediction"].as<double>();
    if (result.count("initial-prediction-error"))
        constants.initialPredictionError = result["initial-prediction-error"].as<double>();
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

    bool isEnvironmentSpecified = (result.count("mux") || result.count("csv"));

    // Show help
    if (result.count("help") || !isEnvironmentSpecified)
    {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }

    uint64_t iterationCount = result["iteration"].as<uint64_t>();
    uint64_t seedCount = result["avg-seeds"].as<uint64_t>();
    uint64_t explorationCount = result["explore"].as<uint64_t>();
    uint64_t exploitationCount = result["exploit"].as<uint64_t>();
    uint64_t smaWidth = result["sma"].as<uint64_t>();

    // Use multiplexer problem
    if (result.count("mux"))
    {
        std::vector<std::unique_ptr<AbstractEnvironment<bool, bool>>> environments;
        for (std::size_t i = 0; i < seedCount; ++i)
        {
            environments.push_back(std::make_unique<MultiplexerEnvironment>(result["mux"].as<int>()));
        }

        run<Experiment<bool, bool>>(
            seedCount,
            { false, true },
            constants,
            iterationCount,
            explorationCount,
            exploitationCount,
            result["coutput"].as<std::string>(),
            result["routput"].as<std::string>(),
            result["noutput"].as<std::string>(),
            smaWidth,
            environments,
            environments);

        exit(0);
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

        std::vector<std::unique_ptr<AbstractEnvironment<int, int>>> explorationEnvironments;
        for (std::size_t i = 0; i < seedCount; ++i)
        {
            explorationEnvironments.push_back(std::make_unique<CSVEnvironment<int, int>>(filename, availableActions, result.count("csv-random")));
        }
        std::vector<std::unique_ptr<AbstractEnvironment<int, int>>> exploitationEnvironments;
        for (std::size_t i = 0; i < seedCount; ++i)
        {
            exploitationEnvironments.push_back(std::make_unique<CSVEnvironment<int, int>>(evaluationCsvFilename, availableActions, result.count("csv-random")));
        }

        run<Experiment<int, int>>(
            seedCount,
            availableActions,
            constants,
            iterationCount,
            explorationCount,
            exploitationCount,
            result["coutput"].as<std::string>(),
            result["routput"].as<std::string>(),
            result["noutput"].as<std::string>(),
            smaWidth,
            explorationEnvironments,
            exploitationEnvironments);

        exit(0);
    }

    // No target environment (show help)
    std::cout << options.help({"", "Group"}) << std::endl;
    return 1;
}