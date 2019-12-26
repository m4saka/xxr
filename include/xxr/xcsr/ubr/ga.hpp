#pragma once

#include <algorithm>

#include "../ga.hpp"

namespace xxr { namespace xcsr_impl { namespace ubr
{

    template <class Population>
    class GA : public xcsr_impl::GA<Population>
    {
    public:
        using typename xcsr_impl::GA<Population>::type;
        using typename xcsr_impl::GA<Population>::SymbolType;
        using typename xcsr_impl::GA<Population>::ConditionType;
        using typename xcsr_impl::GA<Population>::ActionType;
        using typename xcsr_impl::GA<Population>::ConditionActionPairType;
        using typename xcsr_impl::GA<Population>::ConstantsType;
        using typename xcsr_impl::GA<Population>::ClassifierType;
        using typename xcsr_impl::GA<Population>::StoredClassifierType;
        using typename xcsr_impl::GA<Population>::ClassifierPtr;
        using typename xcsr_impl::GA<Population>::ClassifierPtrSetType;
        using typename xcsr_impl::GA<Population>::PopulationType;

    protected:
        using xcsr_impl::GA<Population>::m_constants;
        using xcsr_impl::GA<Population>::m_availableActions;

        // APPLY CROSSOVER (uniform crossover)
        virtual bool uniformCrossover(ClassifierType & cl1, ClassifierType & cl2) const override
        {
            assert(cl1.condition.size() == cl2.condition.size());

            bool isChanged = false;
            for (std::size_t i = 0; i < cl1.condition.size(); ++i)
            {
                if (Random::nextDouble() < 0.5)
                {
                    std::swap(cl1.condition[i].p, cl2.condition[i].p);
                    isChanged = true;
                }
                if (Random::nextDouble() < 0.5)
                {
                    std::swap(cl1.condition[i].q, cl2.condition[i].q);
                    isChanged = true;
                }
            }
            return isChanged;
        }

        // APPLY CROSSOVER (one point crossover)
        virtual bool onePointCrossover(ClassifierType & cl1, ClassifierType & cl2) const override
        {
            assert(cl1.condition.size() == cl2.condition.size());

            std::size_t x = Random::nextInt<std::size_t>(0, cl1.condition.size() * 2);

            bool isChanged = false;
            for (std::size_t i = x + 1; i < cl1.condition.size() * 2; ++i)
            {
                if (i % 2 == 0)
                {
                    std::swap(cl1.condition[i / 2].p, cl2.condition[i / 2].p);
                }
                else
                {
                    std::swap(cl1.condition[i / 2].q, cl2.condition[i / 2].q);
                }
                isChanged = true;
            }
            return isChanged;
        }

        // APPLY CROSSOVER (two point crossover)
        virtual bool twoPointCrossover(ClassifierType & cl1, ClassifierType & cl2) const override
        {
            assert(cl1.condition.size() == cl2.condition.size());

            std::size_t x = Random::nextInt<std::size_t>(0, cl1.condition.size() * 2);
            std::size_t y = Random::nextInt<std::size_t>(0, cl1.condition.size() * 2);

            if (x > y)
            {
                std::swap(x, y);
            }

            bool isChanged = false;
            for (std::size_t i = x + 1; i < y; ++i)
            {
                if (i % 2 == 0)
                {
                    std::swap(cl1.condition[i / 2].p, cl2.condition[i / 2].p);
                }
                else
                {
                    std::swap(cl1.condition[i / 2].q, cl2.condition[i / 2].q);
                }
                isChanged = true;
            }
            return isChanged;
        }

        // APPLY CROSSOVER (BLX-alpha crossover)
        virtual bool blxAlphaCrossover(ClassifierType & cl1, ClassifierType & cl2) const override
        {
            assert(cl1.condition.size() == cl2.condition.size());

            for (std::size_t i = 0; i < cl1.condition.size(); ++i)
            {
                double p1 = cl1.condition[i].p;
                double p2 = cl2.condition[i].p;
                cl1.condition[i].p = p1 + Random::nextDouble(-m_constants.blxAlpha, 1.0 + m_constants.blxAlpha) * (p2 - p1);
                cl2.condition[i].p = p1 + Random::nextDouble(-m_constants.blxAlpha, 1.0 + m_constants.blxAlpha) * (p2 - p1);

                double q1 = cl1.condition[i].q;
                double q2 = cl2.condition[i].q;
                cl1.condition[i].q = q1 + Random::nextDouble(-m_constants.blxAlpha, 1.0 + m_constants.blxAlpha) * (q2 - q1);
                cl2.condition[i].q = q1 + Random::nextDouble(-m_constants.blxAlpha, 1.0 + m_constants.blxAlpha) * (q2 - q1);
            }

            return true;
        }

        // APPLY MUTATION
        virtual void mutate(ClassifierType & cl, const std::vector<type> & situation) const override
        {
            assert(cl.condition.size() == situation.size());

            // Mutate p or q
            for (std::size_t i = 0; i < cl.condition.size(); ++i)
            {
                if (Random::nextDouble() < m_constants.mu)
                {
                    if (Random::nextDouble() < 0.5)
                    {
                        cl.condition[i].p += Random::nextDouble(-m_constants.mutationMaxChange, m_constants.mutationMaxChange);
                        if (m_constants.doRangeRestriction)
                        {
                            cl.condition[i].p = std::min(std::max(m_constants.minValue, cl.condition[i].p), m_constants.maxValue);
                        }
                    }
                    else
                    {
                        cl.condition[i].q += Random::nextDouble(-m_constants.mutationMaxChange, m_constants.mutationMaxChange);
                        if (m_constants.doRangeRestriction)
                        {
                            cl.condition[i].q = std::min(std::max(m_constants.minValue, cl.condition[i].q), m_constants.maxValue);
                        }
                    }
                }
            }

            if (m_constants.doActionMutation && (Random::nextDouble() < m_constants.mu) && (m_availableActions.size() >= 2))
            {
                std::unordered_set<ActionType> otherPossibleActions(m_availableActions);
                otherPossibleActions.erase(cl.action);
                cl.action = Random::chooseFrom(otherPossibleActions);
            }
        }

    public:
        // Constructor
        using xcsr_impl::GA<Population>::GA;

        // Destructor
        virtual ~GA() = default;
    };

}}}
