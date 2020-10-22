#pragma once

#include "../xcs/classifier.hpp"

namespace xxr { namespace xcsr_impl
{

    template <class Condition, typename Action>
    struct ConditionActionPair : public xcs_impl::ConditionActionPair<Condition, Action>
    {
    public:
        using typename xcs_impl::ConditionActionPair<Condition, Action>::type;
        using typename xcs_impl::ConditionActionPair<Condition, Action>::SymbolType;
        using typename xcs_impl::ConditionActionPair<Condition, Action>::ConditionType;
        using typename xcs_impl::ConditionActionPair<Condition, Action>::ActionType;

        using xcs_impl::ConditionActionPair<Condition, Action>::condition;
        using xcs_impl::ConditionActionPair<Condition, Action>::action;

        // Constructor
        using xcs_impl::ConditionActionPair<Condition, Action>::ConditionActionPair;

        // Destructor
        virtual ~ConditionActionPair() = default;

        // IS MORE GENERAL
        virtual bool isMoreGeneral(const xcs_impl::ConditionActionPair<Condition, Action> & cl) const override
        {
            assert(false);
            return false;
        }

        virtual bool isMoreGeneral(const xcs_impl::ConditionActionPair<Condition, Action> & cl, double tolerance) const
        {
            if (&cl == this)
            {
                // Do not subsume itself when tolerance > 0
                return false;
            }

            assert(condition.size() == cl.condition.size());

            std::size_t equalCount = 0;

            for (std::size_t i = 0; i < condition.size(); ++i)
            {
                if (cl.condition.at(i).lower() < condition.at(i).lower() - tolerance || condition.at(i).upper() + tolerance < cl.condition.at(i).upper())
                {
                    return false;
                }

                if (cl.condition.at(i).lower() == condition.at(i).lower() - tolerance && condition.at(i).upper() + tolerance == cl.condition.at(i).upper())
                {
                    ++equalCount;
                }
            }

            if (equalCount == condition.size())
            {
                return false;
            }

            return true;
        }
    };

    template <class Classifier, class Constants>
    struct StoredClassifier : xcs_impl::StoredClassifier<Classifier, Constants>
    {
    public:
        using typename xcs_impl::StoredClassifier<Classifier, Constants>::type;
        using typename xcs_impl::StoredClassifier<Classifier, Constants>::SymbolType;
        using typename xcs_impl::StoredClassifier<Classifier, Constants>::ConditionType;
        using typename xcs_impl::StoredClassifier<Classifier, Constants>::ActionType;
        using typename xcs_impl::StoredClassifier<Classifier, Constants>::ClassifierType;

        using xcs_impl::StoredClassifier<Classifier, Constants>::action;
        using xcs_impl::StoredClassifier<Classifier, Constants>::isSubsumer;
        using xcs_impl::StoredClassifier<Classifier, Constants>::isMoreGeneral;
        using xcs_impl::StoredClassifier<Classifier, Constants>::m_pConstants;

        using xcs_impl::StoredClassifier<Classifier, Constants>::prediction;
        using xcs_impl::StoredClassifier<Classifier, Constants>::epsilon;
        using xcs_impl::StoredClassifier<Classifier, Constants>::fitness;
        using xcs_impl::StoredClassifier<Classifier, Constants>::experience;
        using xcs_impl::StoredClassifier<Classifier, Constants>::timeStamp;
        using xcs_impl::StoredClassifier<Classifier, Constants>::actionSetSize;
        using xcs_impl::StoredClassifier<Classifier, Constants>::numerosity;

        // Constructor
        using xcs_impl::StoredClassifier<Classifier, Constants>::StoredClassifier;

        // Destructor
        virtual ~StoredClassifier() = default;

        // DOES SUBSUME
        virtual bool subsumes(const Classifier & cl) const override
        {
            return action == cl.action && isSubsumer() && isMoreGeneral(cl, m_pConstants->subsumptionTolerance);
        }
    };

}}
