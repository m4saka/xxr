#pragma once
#include <iostream>
#include <fstream>
#include <memory>
#include <functional>
#include <unordered_set>
#include <cstdio>
#include <cstddef>
#include <cmath>
#include "../environment/environment.hpp"
#include "experiment_settings.hpp"
#include "experiment_log_stream.hpp"

namespace xxr
{

    class AbstractExperimentHelper
    {
    public:
        AbstractExperimentHelper() = default;

        virtual ~AbstractExperimentHelper() = default;

        virtual void runIteration(std::size_t repeat = 1) = 0;

        virtual void switchToCondensationMode() = 0;

        virtual void dumpPopulation(std::size_t seedIdx, std::ostream & os) const = 0;
    };

    template <class Experiment, class Environment>
    class ExperimentHelper : public AbstractExperimentHelper
    {
    protected:
        const ExperimentSettings m_settings;
        std::vector<std::unique_ptr<Experiment>> m_experiments;
        std::vector<std::unique_ptr<Environment>> m_explorationEnvironments;
        std::vector<std::unique_ptr<Environment>> m_exploitationEnvironments;
        std::function<void(Environment &)> m_explorationCallback;
        std::function<void(Environment &)> m_exploitationCallback;
        std::ofstream m_summaryLogStream;
        bool m_outputSummaryLogFile;
        SMAExperimentLogStream m_rewardLogStream;
        SMAExperimentLogStream m_systemErrorLogStream;
        SMAExperimentLogStream m_stepCountLogStream;
        ExperimentLogStream m_populationSizeLogStream;
        bool m_alreadyOutputSummaryHeader;
        double m_summaryRewardSum;
        double m_summarySystemErrorSum;
        double m_summaryPopulationSizeSum;
        double m_summaryCoveringOccurrenceRateSum;
        double m_summaryStepCountSum;
        std::size_t m_iterationCount;

        template <class... Args>
        std::vector<std::unique_ptr<Experiment>> makeExperiments(
            const ExperimentSettings & settings,
            const std::unordered_set<typename Experiment::ActionType> & availableActions,
            const typename Experiment::ConstantsType & constants,
            Args && ... args)
        {
            std::vector<std::unique_ptr<Experiment>> experiments;
            for (std::size_t i = 0; i < settings.seedCount; ++i)
            {
                experiments.push_back(
                    std::make_unique<Experiment>(
                        availableActions,
                        constants,
                        std::forward<Args>(args)...
                    )
                );
            }
            return experiments;
        }

        virtual typename Experiment::ActionType callExperimentExploit(std::size_t idx)
        {
            return m_experiments[idx]->exploit(m_exploitationEnvironments[idx]->situation(), m_settings.updateInExploitation);
        }

        virtual typename Experiment::ActionType callExperimentExplore(std::size_t idx)
        {
            return m_experiments[idx]->explore(m_explorationEnvironments[idx]->situation());
        }

        virtual void runExploitationIteration()
        {
            if (m_settings.exploitationCount > 0)
            {
                std::size_t totalStepCount = 0;
                double rewardSum = 0.0;
                double systemErrorSum = 0.0;
                double populationSizeSum = 0.0;
                for (std::size_t j = 0; j < m_settings.seedCount; ++j)
                {
                    for (std::size_t k = 0; k < m_settings.exploitationCount; ++k)
                    {
                        do
                        {
                            // Choose action
                            auto action = callExperimentExploit(j);

                            // Get reward
                            double reward = m_exploitationEnvironments[j]->executeAction(action);
                            m_summaryRewardSum += reward / m_settings.exploitationCount / m_settings.seedCount;
                            m_summarySystemErrorSum += std::abs(reward - m_experiments[j]->prediction()) / m_settings.exploitationCount / m_settings.seedCount;
                            m_summaryCoveringOccurrenceRateSum += static_cast<double>(m_experiments[j]->isCoveringPerformed()) / m_settings.exploitationCount / m_settings.seedCount;
                            if (m_settings.updateInExploitation)
                            {
                                m_experiments[j]->reward(reward, m_exploitationEnvironments[j]->isEndOfProblem());
                            }
                            rewardSum += reward;
                            systemErrorSum += std::abs(reward - m_experiments[j]->prediction());
                            ++totalStepCount;

                            // Run callback if needed
                            m_exploitationCallback(*m_exploitationEnvironments[j]);
                        } while (!m_exploitationEnvironments[j]->isEndOfProblem());

                        populationSizeSum += m_experiments[j]->populationSize();
                    }
                    m_summaryPopulationSizeSum += static_cast<double>(m_experiments[j]->populationSize()) / m_settings.seedCount;
                }

                m_summaryStepCountSum += static_cast<double>(totalStepCount) / m_settings.exploitationCount / m_settings.seedCount;

                if (m_settings.summaryInterval > 0 && (m_iterationCount + 1) % m_settings.summaryInterval == 0)
                {
                    if (!m_alreadyOutputSummaryHeader)
                    {
                        if (m_settings.outputSummaryToStdout)
                        {
                            std::cout
                                << "  Iteration      Reward      SysErr     PopSize  CovOccRate   TotalStep\n"
                                << " ========== =========== =========== =========== =========== ===========" << std::endl;
                        }
                        if (m_summaryLogStream)
                        {
                            m_summaryLogStream << "Iteration,Reward,SysErr,PopSize,CovOccRate,TotalStep" << std::endl;
                        }
                        m_alreadyOutputSummaryHeader = true;
                    }
                    if (m_settings.outputSummaryToStdout)
                    {
                        std::printf("%11u %11.3f %11.3f %11.3f  %1.8f %11.3f\n",
                            static_cast<unsigned int>(m_iterationCount + 1),
                            m_summaryRewardSum / m_settings.summaryInterval,
                            m_summarySystemErrorSum / m_settings.summaryInterval,
                            m_summaryPopulationSizeSum / m_settings.summaryInterval,
                            m_summaryCoveringOccurrenceRateSum / m_settings.summaryInterval,
                            m_summaryStepCountSum / m_settings.summaryInterval);
                        std::fflush(stdout);
                    }
                    if (m_summaryLogStream)
                    {
                        m_summaryLogStream
                            << (m_iterationCount + 1) << ','
                            << m_summaryRewardSum / m_settings.summaryInterval << ','
                            << m_summarySystemErrorSum / m_settings.summaryInterval << ','
                            << m_summaryPopulationSizeSum / m_settings.summaryInterval << ','
                            << m_summaryCoveringOccurrenceRateSum / m_settings.summaryInterval << ','
                            << m_summaryStepCountSum / m_settings.summaryInterval << std::endl;
                    }
                    m_summaryRewardSum = 0.0;
                    m_summarySystemErrorSum = 0.0;
                    m_summaryPopulationSizeSum = 0.0;
                    m_summaryCoveringOccurrenceRateSum = 0.0;
                    m_summaryStepCountSum = 0.0;
                }

                m_rewardLogStream.writeLine(rewardSum / m_settings.exploitationCount / m_settings.seedCount);
                m_systemErrorLogStream.writeLine(systemErrorSum / m_settings.exploitationCount / m_settings.seedCount);
                m_populationSizeLogStream.writeLine(populationSizeSum / m_settings.exploitationCount / m_settings.seedCount);
                m_stepCountLogStream.writeLine(static_cast<double>(totalStepCount) / m_settings.exploitationCount / m_settings.seedCount);
            }
        }

        virtual void runExplorationIteration()
        {
            for (std::size_t j = 0; j < m_settings.seedCount; ++j)
            {
                for (std::size_t k = 0; k < m_settings.explorationCount; ++k)
                {
                    do
                    {
                        // Get situation from environment and choose action
                        auto action = callExperimentExplore(j);

                        // Get reward
                        double reward = m_explorationEnvironments[j]->executeAction(action);
                        m_experiments[j]->reward(reward, m_explorationEnvironments[j]->isEndOfProblem());

                        // Run callback if needed
                        m_explorationCallback(*m_explorationEnvironments[j]);
                    } while (!m_explorationEnvironments[j]->isEndOfProblem());
                }
            }
        }

    public:
        template <class... Args>
        ExperimentHelper(
            const ExperimentSettings & settings,
            const typename Experiment::ConstantsType & constants,
            std::vector<std::unique_ptr<Environment>> && explorationEnvironments,
            std::vector<std::unique_ptr<Environment>> && exploitationEnvironments,
            std::function<void(Environment &)> explorationCallback = [](Environment &){},
            std::function<void(Environment &)> exploitationCallback = [](Environment &){},
            Args && ... args
        )
            : m_settings(settings)
            , m_experiments(
                makeExperiments(
                    settings,
                    dynamic_cast<AbstractEnvironment<typename Experiment::type, typename Experiment::ActionType> &>(*explorationEnvironments.at(0)).availableActions,
                    constants,
                    std::forward<Args>(args)...))
            , m_explorationEnvironments(std::move(explorationEnvironments))
            , m_exploitationEnvironments(std::move(exploitationEnvironments))
            , m_explorationCallback(std::move(explorationCallback))
            , m_exploitationCallback(std::move(exploitationCallback))
            , m_summaryLogStream(settings.outputSummaryFilename.empty() ? "" : (settings.outputFilenamePrefix + settings.outputSummaryFilename))
            , m_rewardLogStream(settings.outputRewardFilename.empty() ? "" : (settings.outputFilenamePrefix + settings.outputRewardFilename), settings.smaWidth, false)
            , m_systemErrorLogStream(settings.outputSystemErrorFilename.empty() ? "" : (settings.outputFilenamePrefix + settings.outputSystemErrorFilename), settings.smaWidth, false)
            , m_stepCountLogStream(settings.outputStepCountFilename.empty() ? "" : (settings.outputFilenamePrefix + settings.outputStepCountFilename), settings.smaWidth, false)
            , m_populationSizeLogStream(settings.outputPopulationSizeFilename.empty() ? "" : (settings.outputFilenamePrefix + settings.outputPopulationSizeFilename), false)
            , m_alreadyOutputSummaryHeader(false)
            , m_summaryRewardSum(0.0)
            , m_summarySystemErrorSum(0.0)
            , m_summaryPopulationSizeSum(0.0)
            , m_summaryCoveringOccurrenceRateSum(0.0)
            , m_summaryStepCountSum(0.0)
            , m_iterationCount(0)
        {
            if (!settings.inputClassifierFilename.empty())
            {
                for (auto && experiment : m_experiments)
                {
                    experiment->loadPopulationCSV(settings.inputClassifierFilename, !settings.useInputClassifierToResume);
                }
            }
        }

        virtual ~ExperimentHelper() = default;

        virtual void runIteration(std::size_t repeat = 1) override
        {
            for (std::size_t i = 0; i < repeat; ++i)
            {
                runExploitationIteration();
                runExplorationIteration();
                ++m_iterationCount;
            }
        }

        virtual void switchToCondensationMode() override
        {
            for (auto && experiment : m_experiments)
            {
                experiment->switchToCondensationMode();
            }
        }

        std::size_t seedCount() const noexcept
        {
            return m_settings.seedCount;
        }

        Experiment & experimentAt(std::size_t seedIdx)
        {
            return *m_experiments[seedIdx];
        }

        Environment & explorationEnvironmentAt(std::size_t seedIdx)
        {
            return *m_explorationEnvironments[seedIdx];
        }

        Environment & exploitationEnvironmentAt(std::size_t seedIdx)
        {
            return *m_exploitationEnvironments[seedIdx];
        }
        
        virtual void dumpPopulation(std::size_t seedIdx, std::ostream & os) const override
        {
            m_experiments[seedIdx]->dumpPopulation(os);
        }
    };

}
