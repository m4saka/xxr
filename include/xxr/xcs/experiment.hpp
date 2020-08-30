#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>

#include "../experiment.hpp"
#include "constants.hpp"
#include "symbol.hpp"
#include "condition.hpp"
#include "classifier.hpp"
#include "classifier_ptr_set.hpp"
#include "population.hpp"
#include "match_set.hpp"
#include "action_set.hpp"
#include "ga.hpp"
#include "prediction_array.hpp"
#include "../random.hpp"
#include "../helper/csv.hpp"

namespace xxr { namespace xcs_impl
{

    template <
        typename T,
        typename Action,
        class PredictionArray = EpsilonGreedyPredictionArray<
            MatchSet<
                Population<
                    ClassifierPtrSet<
                        StoredClassifier<
                            Classifier<ConditionActionPair<Condition<Symbol<T>>, Action>>,
                            Constants
                        >
                    >
                >
            >
        >,
        class ActionSet = ActionSet<
            GA<
                Population<
                    ClassifierPtrSet<
                        StoredClassifier<
                            Classifier<ConditionActionPair<Condition<Symbol<T>>, Action>>,
                            Constants
                        >
                    >
                >
            >
        >
    >
    class Experiment : public AbstractExperiment<T, Action>
    {
    public:
        static_assert(
            std::is_same<typename PredictionArray::PopulationType, typename ActionSet::PopulationType>::value,
            "PredictionArray::PopulationType and ActionSet::PopulationType must not be different");
        static_assert(
            std::is_same<T, typename PredictionArray::type>::value,
            "Experiment::type and PredictionArray::type must not be different");
        static_assert(
            std::is_same<T, typename ActionSet::type>::value,
            "Experiment::type and ActionSet::type must not be different");
        static_assert(
            std::is_same<Action, typename PredictionArray::ActionType>::value,
            "Experiment::ActionType and PredictionArray::ActionType must not be different");
        static_assert(
            std::is_same<Action, typename ActionSet::ActionType>::value,
            "Experiment::ActionType and ActionSet::ActionType must not be different");

        using type = T;
        using SymbolType = typename PredictionArray::SymbolType;
        using ConditionType = typename PredictionArray::ConditionType;
        using ActionType = Action;
        using ConditionActionPairType = typename PredictionArray::ConditionActionPairType;
        using ConstantsType = typename PredictionArray::ConstantsType;
        using ClassifierType = typename PredictionArray::ClassifierType;
        using StoredClassifierType = typename PredictionArray::StoredClassifierType;
        using ClassifierPtr = typename PredictionArray::ClassifierPtr;
        using ClassifierPtrSetType = typename PredictionArray::ClassifierPtrSetType;
        using PopulationType = typename PredictionArray::PopulationType;
        using MatchSetType = typename PredictionArray::MatchSetType;
        using PredictionArrayType = PredictionArray;
        using GAType = typename ActionSet::GAType;
        using ActionSetType = ActionSet;

        ConstantsType constants;

    protected:
        // [P]
        //   The population [P] consists of all classifier that exist in XCS at any time.
        PopulationType m_population;

        // [A]
        //   The action set [A] is formed out of the current [M].
        //   It includes all classifiers of [M] that propose the executed action.
        ActionSet m_actionSet;

        // [A]_-1
        //   The previous action set [A]_-1 is the action set that was active in the last
        //   execution cycle.
        ActionSet m_prevActionSet;

        // Available action choices
        std::unordered_set<Action> m_availableActions;

        uint64_t m_timeStamp;

        bool m_expectsReward;
        double m_prevReward;
        bool m_isPrevModeExplore;

        std::vector<T> m_prevSituation;

        // Prediction value of the previous action decision (just for logging)
        double m_prediction;
        std::unordered_map<int, double> m_predictions;

        // Covering occurrence of the previous action decision (just for logging)
        bool m_isCoveringPerformed;

    public:
        // Constructor
        Experiment(const std::unordered_set<Action> & availableActions, const ConstantsType & constants)
            : constants(constants)
            , m_population(&this->constants, availableActions)
            , m_actionSet(&this->constants, availableActions)
            , m_prevActionSet(&this->constants, availableActions)
            , m_availableActions(availableActions)
            , m_timeStamp(0)
            , m_expectsReward(false)
            , m_prevReward(0.0)
            , m_isPrevModeExplore(false)
            , m_prediction(0.0)
            , m_isCoveringPerformed(false)
        {
        }

        // Destructor
        virtual ~Experiment() = default;

        // Run with exploration
        virtual Action explore(const std::vector<T> & situation) override
        {
            assert(!m_expectsReward);

            // [M]
            //   The match set [M] is formed out of the current [P].
            //   It includes all classifiers that match the current situation.
            const MatchSetType matchSet(m_population, situation, m_timeStamp, &this->constants, m_availableActions);
            m_isCoveringPerformed = matchSet.isCoveringPerformed();

            const PredictionArray predictionArray(matchSet, this->constants.exploreProbability);

            const Action action = predictionArray.selectAction();
            m_prediction = predictionArray.predictionFor(action);
            for (const auto & action : m_availableActions)
            {
                m_predictions[action] = predictionArray.predictionFor(action);
            }

            m_actionSet.regenerate(matchSet, action);

            m_expectsReward = true;
            m_isPrevModeExplore = true;

            if (!m_prevActionSet.empty())
            {
                double p = m_prevReward + constants.gamma * predictionArray.max();
                m_prevActionSet.update(p, m_population);
                m_prevActionSet.runGA(m_prevSituation, m_population, m_timeStamp);
            }

            m_prevSituation = situation;

            return action;
        }

        virtual void reward(double value, bool isEndOfProblem = true) override
        {
            assert(m_expectsReward);

            if (isEndOfProblem)
            {
                m_actionSet.update(value, m_population);
                if (m_isPrevModeExplore) // Do not perform GA operations in exploitation
                {
                    m_actionSet.runGA(m_prevSituation, m_population, m_timeStamp);
                }
                m_prevActionSet.clear();
            }
            else
            {
                m_actionSet.copyTo(m_prevActionSet);
                m_prevReward = value;
            }

            if (m_isPrevModeExplore) // Do not increment actual time in exploitation
            {
                ++m_timeStamp;
            }

            m_expectsReward = false;
        }

        // Run without exploration
        // (Set update to true when testing multi-step problems. If update is true, make sure to call reward() after this.)
        virtual Action exploit(const std::vector<T> & situation, bool update = false) override
        {
            if (update)
            {
                assert(!m_expectsReward);

                // [M]
                //   The match set [M] is formed out of the current [P].
                //   It includes all classifiers that match the current situation.
                const MatchSetType matchSet(m_population, situation, m_timeStamp, &this->constants, m_availableActions);
                m_isCoveringPerformed = matchSet.isCoveringPerformed();

                const GreedyPredictionArray<MatchSetType> predictionArray(matchSet);

                const Action action = predictionArray.selectAction();

                m_actionSet.regenerate(matchSet, action);

                m_expectsReward = true;
                m_isPrevModeExplore = false;

                if (!m_prevActionSet.empty())
                {
                    double p = m_prevReward + this->constants.gamma * predictionArray.max();
                    m_prevActionSet.update(p, m_population);

                    // Do not perform GA operations in exploitation
                }

                m_prevSituation = situation;

                return action;
            }
            else
            {
                // Create new match set as sandbox
                MatchSetType matchSet(&this->constants, m_availableActions);
                for (auto && cl : m_population)
                {
                    if (cl->condition.matches(situation))
                    {
                        matchSet.insert(cl);
                    }
                }

                if (!matchSet.empty())
                {
                    m_isCoveringPerformed = false;

                    GreedyPredictionArray<MatchSetType> predictionArray(matchSet);
                    const Action action = predictionArray.selectAction();
                    m_prediction = predictionArray.predictionFor(action);
                    for (const auto & action : m_availableActions)
                    {
                        m_predictions[action] = predictionArray.predictionFor(action);
                    }
                    return action;
                }
                else
                {
                    m_isCoveringPerformed = true;
                    m_prediction = this->constants.initialPrediction;
                    for (const auto & action : m_availableActions)
                    {
                        m_predictions[action] = this->constants.initialPrediction;
                    }
                    return Random::chooseFrom(m_availableActions);
                }
            }
        }

        // Get prediction value of the previous action decision
        // (Call this function after explore() or exploit())
        virtual double prediction() const
        {
            return m_prediction;
        }

        // Get prediction value of the action
        // (Call this function after explore() or exploit())
        double predictionFor(int action) const
        {
            return m_predictions.at(action);
        }

        // Get if covering is performed in the previous action decision
        // (Call this function after explore() or exploit())
        virtual bool isCoveringPerformed() const
        {
            return m_isCoveringPerformed;
        }

        virtual std::vector<ClassifierType> getMatchingClassifiers(const std::vector<T> & situation) const
        {
            std::vector<ClassifierType> classifiers;
            for (auto && cl : m_population)
            {
                if (cl->condition.matches(situation))
                {
                    classifiers.emplace_back(*cl);
                }
            }
            return classifiers;
        }

        virtual void loadPopulationCSV(const std::string & filename, bool useAsInitialPopulation = true) override
        {
            auto population = xxr::CSV::readPopulation<ClassifierType>(filename);
            if (useAsInitialPopulation)
            {
                for (auto && cl : population)
                {
                    cl.prediction = this->constants.initialPrediction;
                    cl.epsilon = this->constants.initialEpsilon;
                    cl.fitness = this->constants.initialFitness;
                    cl.experience = 0;
                    cl.timeStamp = 0;
                    cl.actionSetSize = 1;
                    //cl.numerosity = 1; // commented out to keep macroclassifier as is
                }
            }
            setPopulation(population, !useAsInitialPopulation);
        }

        virtual void setPopulation(const std::vector<ClassifierType> & classifiers, bool initTimeStamp = true)
        {
            // Replace population
            m_population.clear();
            for (auto && cl : classifiers)
            {
                m_population.emplace(std::make_shared<StoredClassifierType>(cl, &this->constants));
            }

            // Clear action set and reset status
            m_actionSet.clear();
            m_prevActionSet.clear();
            m_expectsReward = false;
            m_isPrevModeExplore = false;

            // Set system timestamp to the same as latest classifier
            if (initTimeStamp)
            {
                m_timeStamp = 0;
                for (auto && cl : classifiers)
                {
                    if (m_timeStamp < cl.timeStamp)
                    {
                        m_timeStamp = cl.timeStamp;
                    }
                }
            }
        }

        virtual void dumpPopulation(std::ostream & os) const override
        {
            os << "Condition,Action,prediction,epsilon,F,exp,ts,as,n,acc\n";
            for (auto && cl : m_population)
            {
                os  << cl->condition << ","
                    << cl->action << ","
                    << cl->prediction << ","
                    << cl->epsilon << ","
                    << cl->fitness << ","
                    << cl->experience << ","
                    << cl->timeStamp << ","
                    << cl->actionSetSize << ","
                    << cl->numerosity << ","
                    << cl->accuracy() << "\n";
            }
        }

        virtual std::size_t populationSize() const noexcept override
        {
            return m_population.size();
        }

        virtual std::size_t numerositySum() const override
        {
            uint64_t sum = 0;
            for (auto && cl : m_population)
            {
                sum += cl->numerosity;
            }
            return sum;
        }

        virtual void switchToCondensationMode() noexcept override
        {
            constants.chi = 0.0;
            constants.mu = 0.0;
        }
    };

}}
