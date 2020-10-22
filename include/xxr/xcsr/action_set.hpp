#pragma once

#include "../xcs/action_set.hpp"

namespace xxr { namespace xcsr_impl
{

    template <class GA>
    class ActionSet : public xcs_impl::ActionSet<GA>
    {
    public:
        using typename xcs_impl::ActionSet<GA>::type;
        using typename xcs_impl::ActionSet<GA>::SymbolType;
        using typename xcs_impl::ActionSet<GA>::ConditionType;
        using typename xcs_impl::ActionSet<GA>::ActionType;
        using typename xcs_impl::ActionSet<GA>::ConditionActionPairType;
        using typename xcs_impl::ActionSet<GA>::ConstantsType;
        using typename xcs_impl::ActionSet<GA>::ClassifierType;
        using typename xcs_impl::ActionSet<GA>::StoredClassifierType;
        using typename xcs_impl::ActionSet<GA>::ClassifierPtr;
        using typename xcs_impl::ActionSet<GA>::ClassifierPtrSetType;
        using typename xcs_impl::ActionSet<GA>::PopulationType;
        using typename xcs_impl::ActionSet<GA>::GAType;

    protected:
        using xcs_impl::ActionSet<GA>::m_set;
        using xcs_impl::ActionSet<GA>::m_pConstants;
        using xcs_impl::ActionSet<GA>::m_availableActions;

        // DO ACTION SET SUBSUMPTION
        virtual void doSubsumption(PopulationType & population) override
        {
            ClassifierPtr cl;

            for (auto && c : m_set)
            {
                if (c->isSubsumer())
                {
                    if (cl.get() == nullptr || c->isMoreGeneral(*cl, m_pConstants->subsumptionTolerance))
                    {
                        cl = c;
                    }
                }
            }

            if (cl.get() != nullptr)
            {
                std::vector<ClassifierPtr> removedClassifiers;
                for (auto && c : m_set)
                {
                    if (cl->isMoreGeneral(*c, m_pConstants->subsumptionTolerance))
                    {
                        cl->numerosity += c->numerosity;
                        removedClassifiers.push_back(c);
                    }
                }

                for (auto && removedClassifier : removedClassifiers)
                {
                    population.erase(removedClassifier);
                    m_set.erase(removedClassifier);
                }
            }
        }

    public:
        // Constructor
        ActionSet(const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions) :
            xcs_impl::ActionSet<GA>(pConstants, availableActions)
        {
        }

        template <class MatchSet>
        ActionSet(const MatchSet & matchSet, ActionType action, const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions) :
            ActionSet(pConstants, availableActions)
        {
            this->regenerate(matchSet, action);
        }

        // Destructor
        virtual ~ActionSet() = default;
    };

}}
