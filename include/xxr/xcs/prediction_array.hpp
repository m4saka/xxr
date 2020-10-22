#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <random>
#include <limits>
#include <cfloat>
#include <cmath>

namespace xxr { namespace xcs_impl
{

    template <class MatchSet>
    class AbstractPredictionArray
    {
    public:
        using type = typename MatchSet::type;
        using SymbolType = typename MatchSet::SymbolType;
        using ConditionType = typename MatchSet::ConditionType;
        using ActionType = typename MatchSet::ActionType;
        using ConditionActionPairType = typename MatchSet::ConditionActionPairType;
        using ConstantsType = typename MatchSet::ConstantsType;
        using ClassifierType = typename MatchSet::ClassifierType;
        using StoredClassifierType = typename MatchSet::StoredClassifierType;
        using ClassifierPtr = typename MatchSet::ClassifierPtr;
        using ClassifierPtrSetType = typename MatchSet::ClassifierPtrSetType;
        using PopulationType = typename MatchSet::PopulationType;
        using MatchSetType = MatchSet;

    protected:
        const ConstantsType * const m_pConstants;

        // PA (Prediction Array)
        std::unordered_map<ActionType, double> m_pa;

        // Array of PA keys (for random action selection)
        std::vector<ActionType> m_paActions;

        // The maximum value of PA
        double m_maxPA;

        // The best actions of PA
        std::vector<ActionType> m_maxPAActions;

    public:
        // GENERATE PREDICTION ARRAY
        AbstractPredictionArray(const MatchSet & matchSet, const ConstantsType *pConstants)
            : m_pConstants(pConstants)
        {
            // FSA (Fitness Sum Array)
            std::unordered_map<ActionType, double> fsa;

            for (auto && cl : matchSet)
            {
                if (m_pa.count(cl->action) == 0) {
                    m_paActions.push_back(cl->action);
                }
                m_pa[cl->action] += cl->prediction * cl->fitness;
                fsa[cl->action] += cl->fitness;
            }

            m_maxPA = -100000.0;

            for (auto && pair : m_pa)
            {
                if (std::abs(fsa[pair.first]) > 0.0)
                {
                    pair.second /= fsa[pair.first];
                }

                // Update the best actions
                if (std::abs(m_maxPA - pair.second) < DBL_EPSILON) // m_maxPA == pair.second
                {
                    m_maxPAActions.push_back(pair.first);
                }
                else if (m_maxPA < pair.second)
                {
                    m_maxPAActions.clear();
                    m_maxPAActions.push_back(pair.first);
                    m_maxPA = pair.second;
                }
            }
        }

        // Destructor
        virtual ~AbstractPredictionArray() = default;

        virtual double max() const
        {
            assert(m_maxPA != -100000.0);
            return m_maxPA;
        }

        virtual double predictionFor(ActionType action) const
        {
            return m_pa.count(action) ? m_pa.at(action) : m_pConstants->initialPrediction;
        }

        // SELECT ACTION
        virtual ActionType selectAction() const = 0;
    };

    template <class MatchSet>
    class GreedyPredictionArray final : public AbstractPredictionArray<MatchSet>
    {
    public:
        using typename AbstractPredictionArray<MatchSet>::type;
        using typename AbstractPredictionArray<MatchSet>::SymbolType;
        using typename AbstractPredictionArray<MatchSet>::ConditionType;
        using typename AbstractPredictionArray<MatchSet>::ActionType;
        using typename AbstractPredictionArray<MatchSet>::ConstantsType;
        using typename AbstractPredictionArray<MatchSet>::ClassifierType;
        using typename AbstractPredictionArray<MatchSet>::StoredClassifierType;
        using typename AbstractPredictionArray<MatchSet>::ClassifierPtr;
        using typename AbstractPredictionArray<MatchSet>::ClassifierPtrSetType;
        using typename AbstractPredictionArray<MatchSet>::PopulationType;

    private:
        using AbstractPredictionArray<MatchSet>::m_maxPAActions;

    public:
        // Constructor
        using AbstractPredictionArray<MatchSet>::AbstractPredictionArray;

        ActionType selectAction() const override
        {
            // Choose best action
            assert(!m_maxPAActions.empty());
            return Random::chooseFrom(m_maxPAActions);
        }
    };

    template <class MatchSet>
    class EpsilonGreedyPredictionArray final : public AbstractPredictionArray<MatchSet>
    {
    public:
        using typename AbstractPredictionArray<MatchSet>::type;
        using typename AbstractPredictionArray<MatchSet>::SymbolType;
        using typename AbstractPredictionArray<MatchSet>::ConditionType;
        using typename AbstractPredictionArray<MatchSet>::ActionType;
        using typename AbstractPredictionArray<MatchSet>::ConstantsType;
        using typename AbstractPredictionArray<MatchSet>::ClassifierType;
        using typename AbstractPredictionArray<MatchSet>::StoredClassifierType;
        using typename AbstractPredictionArray<MatchSet>::ClassifierPtrSetType;
        using typename AbstractPredictionArray<MatchSet>::PopulationType;

    private:
        const double m_epsilon;
        using AbstractPredictionArray<MatchSet>::m_paActions;
        using AbstractPredictionArray<MatchSet>::m_maxPAActions;

    public:
        // Constructor
        EpsilonGreedyPredictionArray(const MatchSet & matchSet, const ConstantsType *pConstants, double epsilon)
            : AbstractPredictionArray<MatchSet>(matchSet, pConstants), m_epsilon(epsilon) {}

        ActionType selectAction() const override
        {
            if (Random::nextDouble() < m_epsilon)
            {
                assert(!m_paActions.empty());
                return Random::chooseFrom(m_paActions); // Choose random action
            }
            else
            {
                assert(!m_maxPAActions.empty());
                return Random::chooseFrom(m_maxPAActions); // Choose best action
            }
        }
    };

}}
