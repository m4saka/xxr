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
        using typename xcs_impl::ActionSet<GA>::ClassifierPtrSetType;
        using typename xcs_impl::ActionSet<GA>::PopulationType;
        using typename xcs_impl::ActionSet<GA>::GAType;

    protected:
        using ClassifierPtr = std::shared_ptr<StoredClassifierType>;
        using xcs_impl::ActionSet<GA>::m_set;
        using xcs_impl::ActionSet<GA>::m_constants;
        using xcs_impl::ActionSet<GA>::m_availableActions;

        // DO ACTION SET SUBSUMPTION
        virtual void doSubsumption(PopulationType & population) override
        {
            ClassifierPtr cl;

            for (auto && c : m_set)
            {
                if (c->isSubsumer())
                {
                    if (cl.get() == nullptr || c->isMoreGeneral(*cl, m_constants.subsumptionTolerance))
                    {
                        cl = c;
                    }
                }
            }

            if (cl.get() != nullptr)
            {
                std::vector<const ClassifierPtr *> removedClassifiers;
                for (auto && c : m_set)
                {
                    if (cl->isMoreGeneral(*c, m_constants.subsumptionTolerance))
                    {
                        cl->numerosity += c->numerosity;
                        removedClassifiers.push_back(&c);
                    }
                }

                for (auto && removedClassifier : removedClassifiers)
                {
                    population.erase(*removedClassifier);
                    m_set.erase(*removedClassifier);
                }
            }
        }

    public:
        // Constructor
        ActionSet(ConstantsType & constants, const std::unordered_set<ActionType> & availableActions) :
            xcs_impl::ActionSet<GA>(constants, availableActions)
        {
        }

        template <class MatchSet>
        ActionSet(const MatchSet & matchSet, ActionType action, ConstantsType & constants, const std::unordered_set<ActionType> & availableActions) :
            ActionSet(constants, availableActions)
        {
            this->regenerate(matchSet, action);
        }

        // Destructor
        virtual ~ActionSet() = default;
    };

}}
