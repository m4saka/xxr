#pragma once

#include <algorithm>

#include "../ga.hpp"

namespace xxr { namespace xcsr_impl { namespace csr
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
        using xcsr_impl::GA<Population>::m_pConstants;
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
                    std::swap(cl1.condition[i].center, cl2.condition[i].center);
                    isChanged = true;
                }
                if (Random::nextDouble() < 0.5)
                {
                    std::swap(cl1.condition[i].spread, cl2.condition[i].spread);
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
                    std::swap(cl1.condition[i / 2].center, cl2.condition[i / 2].center);
                }
                else
                {
                    std::swap(cl1.condition[i / 2].spread, cl2.condition[i / 2].spread);
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
                    std::swap(cl1.condition[i / 2].center, cl2.condition[i / 2].center);
                }
                else
                {
                    std::swap(cl1.condition[i / 2].spread, cl2.condition[i / 2].spread);
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
                double c1 = cl1.condition[i].center;
                double c2 = cl2.condition[i].center;
                cl1.condition[i].center = c1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (c2 - c1);
                cl2.condition[i].center = c1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (c2 - c1);

                double s1 = cl1.condition[i].spread;
                double s2 = cl2.condition[i].spread;
                cl1.condition[i].spread = s1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (s2 - s1);
                cl2.condition[i].spread = s1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (s2 - s1);
            }

            return true;
        }

        // APPLY MUTATION
        virtual void mutate(ClassifierType & cl, const std::vector<type> & situation) const override
        {
            assert(cl.condition.size() == situation.size());

            // Mutate center or spread
            for (std::size_t i = 0; i < cl.condition.size(); ++i)
            {
                if (Random::nextDouble() < m_pConstants->mu)
                {
                    if (Random::nextDouble() < 0.5)
                    {
                        cl.condition[i].center += Random::nextDouble(-m_pConstants->mutationMaxChange, m_pConstants->mutationMaxChange);
                        cl.condition[i].center = std::min(std::max(m_pConstants->minValue, cl.condition[i].center), m_pConstants->maxValue);
                    }
                    else
                    {
                        cl.condition[i].spread += Random::nextDouble(-m_pConstants->mutationMaxChange, m_pConstants->mutationMaxChange);
                        cl.condition[i].spread = std::max(0.0, cl.condition[i].spread);
                    }
                }
            }

            if (m_pConstants->doActionMutation && (Random::nextDouble() < m_pConstants->mu) && (m_availableActions.size() >= 2))
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
