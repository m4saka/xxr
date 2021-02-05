#pragma once

#include "../../xcs/experiment.hpp"
#include "../condition.hpp"
#include "symbol.hpp"
#include "../classifier.hpp"
#include "../constants.hpp"
#include "match_set.hpp"
#include "ga.hpp"
#include "../action_set.hpp"

namespace xxr { namespace xcsr_impl { namespace ubr
{

    template <
        typename T,
        typename Action,
        class PredictionArray = xcs_impl::EpsilonGreedyPredictionArray<
            MatchSet<
                xcs_impl::Population<
                    xcs_impl::ClassifierPtrSet<
                        xcsr_impl::StoredClassifier<
                            xcs_impl::Classifier<xcsr_impl::ConditionActionPair<xcsr_impl::Condition<Symbol<T>>, Action>>,
                            xcsr_impl::Constants
                        >
                    >
                >
            >
        >,
        class ActionSet = xcsr_impl::ActionSet<
            GA<
                xcs_impl::Population<
                    xcs_impl::ClassifierPtrSet<
                        xcsr_impl::StoredClassifier<
                            xcs_impl::Classifier<xcsr_impl::ConditionActionPair<xcsr_impl::Condition<Symbol<T>>, Action>>,
                            xcsr_impl::Constants
                        >
                    >
                >
            >
        >
    >
    class Experiment : public xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>
    {
    public:
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::type;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::SymbolType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ConditionType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ActionType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ConditionActionPairType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ConstantsType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ClassifierType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::StoredClassifierType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ClassifierPtr;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ClassifierPtrSetType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::PopulationType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::MatchSetType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::PredictionArrayType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::GAType;
        using typename xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::ActionSetType;

        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::constants;

        // Constructor
        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::Experiment;

        // Destructor
        virtual ~Experiment() = default;

        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::explore;
        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::reward;
        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::exploit;

        virtual void loadPopulationCSV(const std::string & filename, bool useAsInitialPopulation = true) override
        {
            auto population = xxr::CSV::readPopulation<ClassifierType>(filename, true, true /* skip first column only for XCSR population */);
            if (useAsInitialPopulation)
            {
                for (auto && cl : population)
                {
                    cl.prediction = constants.initialPrediction;
                    cl.epsilon = constants.initialEpsilon;
                    cl.fitness = constants.initialFitness;
                    cl.experience = 0;
                    cl.timeStamp = 0;
                    cl.actionSetSize = 1;
                    //cl.numerosity = 1; // commented out to keep macroclassifier as is
                }
            }
            setPopulation(population, useAsInitialPopulation);
        }

        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::population;

        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::setPopulation;

        virtual void dumpPopulation(std::ostream & os) const override
        {
            os  << "Condition[" << constants.minValue << "-" << constants.maxValue << "],"
                << "Condition[p;q],Action,prediction,epsilon,F,exp,ts,as,n,acc" << std::endl;

            for (auto && cl : this->m_population)
            {
                for (auto && symbol : cl->condition)
                {
                    os << "|";

                    auto normalizedLowerLimit = (symbol.lower() - constants.minValue) / (constants.maxValue - constants.minValue);
                    auto normalizedUpperLimit = (symbol.upper() - constants.minValue) / (constants.maxValue - constants.minValue);

                    for (int i = 0; i < 10; ++i)
                    {
                        if (normalizedLowerLimit < i / 10.0 && (i + 1) / 10.0 < normalizedUpperLimit)
                        {
                            os << "O";
                        }
                        else if ((i / 10.0 <= normalizedLowerLimit && normalizedLowerLimit <= (i + 1) / 10.0)
                            || (i / 10.0 <= normalizedUpperLimit && normalizedUpperLimit <= (i + 1) / 10.0))
                        {
                            os << "o";
                        }
                        else
                        {
                            os << ".";
                        }
                    }
                }
                os << "|" << ",";

                os  << cl->condition << ","
                    << cl->action << ","
                    << cl->prediction << ","
                    << cl->epsilon << ","
                    << cl->fitness << ","
                    << cl->experience << ","
                    << cl->timeStamp << ","
                    << cl->actionSetSize << ","
                    << cl->numerosity << ","
                    << cl->accuracy() << std::endl;
            }
        }

        virtual void switchToCondensationMode() noexcept override
        {
            constants.chi = 0.0;
            constants.mu = 0.0;
            constants.subsumptionTolerance = 0.0;
        }

        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::populationSize;
        using xcs_impl::Experiment<T, Action, PredictionArray, ActionSet>::numerositySum;
    };

}}}
