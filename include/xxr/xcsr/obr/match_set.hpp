#pragma once

#include "../../xcs/match_set.hpp"

namespace xxr { namespace xcsr_impl { namespace obr
{

    template <class Population>
    class MatchSet : public xcs_impl::MatchSet<Population>
    {
    public:
        using typename xcs_impl::MatchSet<Population>::type;
        using typename xcs_impl::MatchSet<Population>::SymbolType;
        using typename xcs_impl::MatchSet<Population>::ConditionType;
        using typename xcs_impl::MatchSet<Population>::ActionType;
        using typename xcs_impl::MatchSet<Population>::ConditionActionPairType;
        using typename xcs_impl::MatchSet<Population>::ConstantsType;
        using typename xcs_impl::MatchSet<Population>::ClassifierType;
        using typename xcs_impl::MatchSet<Population>::StoredClassifierType;
        using typename xcs_impl::MatchSet<Population>::ClassifierPtr;
        using typename xcs_impl::MatchSet<Population>::ClassifierPtrSetType;
        using typename xcs_impl::MatchSet<Population>::PopulationType;

    protected:
        using xcs_impl::MatchSet<Population>::m_set;
        using xcs_impl::MatchSet<Population>::m_pConstants;
        using xcs_impl::MatchSet<Population>::m_availableActions;

        // GENERATE COVERING CLASSIFIER
        virtual ClassifierPtr generateCoveringClassifier(const std::vector<type> & situation, const std::unordered_set<ActionType> & unselectedActions, uint64_t timeStamp) const override
        {
            std::vector<SymbolType> symbols;
            for (auto && symbol : situation)
            {
                double lowerMin = symbol - m_pConstants->coveringMaxSpread;
                double upperMax = symbol + m_pConstants->coveringMaxSpread;
                if (m_pConstants->doCoveringRandomRangeTruncation)
                {
                    lowerMin = std::max(lowerMin, m_pConstants->minValue);
                    upperMax = std::min(upperMax, m_pConstants->maxValue);
                }

                double lower = Random::nextDouble(lowerMin, symbol);
                double upper = Random::nextDouble(symbol, upperMax);
                if (m_pConstants->doRangeRestriction)
                {
                    lower = std::max(lower, m_pConstants->minValue);
                    upper = std::min(upper, m_pConstants->maxValue);
                }

                symbols.emplace_back(lower, upper);
            }

            return std::make_shared<StoredClassifierType>(symbols, Random::chooseFrom(unselectedActions), timeStamp, m_pConstants);
        }

    public:
        // Constructor
        MatchSet(const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : xcs_impl::MatchSet<Population>(pConstants, availableActions)
        {
        }

        MatchSet(Population & population, const std::vector<type> & situation, uint64_t timeStamp, const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : MatchSet(pConstants, availableActions)
        {
            this->regenerate(population, situation, timeStamp);
        }

        // Destructor
        virtual ~MatchSet() = default;
    };

}}}
