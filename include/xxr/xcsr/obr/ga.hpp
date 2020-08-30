#pragma once

#include <algorithm>

#include "../ga.hpp"

namespace xxr { namespace xcsr_impl { namespace obr
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
                    std::swap(cl1.condition[i].l, cl2.condition[i].l);
                    isChanged = true;
                }
                if (Random::nextDouble() < 0.5)
                {
                    std::swap(cl1.condition[i].u, cl2.condition[i].u);
                    isChanged = true;
                }
            }

            // Fix lower and upper order
            for (std::size_t i = 0; i < cl1.condition.size(); ++i)
            {
                if (cl1.condition[i].l > cl1.condition[i].u)
                {
                    std::swap(cl1.condition[i].l, cl1.condition[i].u);
                }

                if (cl2.condition[i].l > cl2.condition[i].u)
                {
                    std::swap(cl2.condition[i].l, cl2.condition[i].u);
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
                    std::swap(cl1.condition[i / 2].l, cl2.condition[i / 2].l);
                }
                else
                {
                    std::swap(cl1.condition[i / 2].u, cl2.condition[i / 2].u);
                }
                isChanged = true;
            }

            // Fix lower and upper order
            for (std::size_t i = (x + 1) / 2; i < cl1.condition.size(); ++i)
            {
                if (cl1.condition[i].l > cl1.condition[i].u)
                {
                    std::swap(cl1.condition[i].l, cl1.condition[i].u);
                }

                if (cl2.condition[i].l > cl2.condition[i].u)
                {
                    std::swap(cl2.condition[i].l, cl2.condition[i].u);
                }
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
                    std::swap(cl1.condition[i / 2].l, cl2.condition[i / 2].l);
                }
                else
                {
                    std::swap(cl1.condition[i / 2].u, cl2.condition[i / 2].u);
                }
                isChanged = true;
            }

            // Fix lower and upper order
            for (std::size_t i = (x + 1) / 2; i < y / 2; ++i)
            {
                if (cl1.condition[i].l > cl1.condition[i].u)
                {
                    std::swap(cl1.condition[i].l, cl1.condition[i].u);
                }

                if (cl2.condition[i].l > cl2.condition[i].u)
                {
                    std::swap(cl2.condition[i].l, cl2.condition[i].u);
                }
            }

            return isChanged;
        }

        // APPLY CROSSOVER (BLX-alpha crossover)
        virtual bool blxAlphaCrossover(ClassifierType & cl1, ClassifierType & cl2) const override
        {
            assert(cl1.condition.size() == cl2.condition.size());

            for (std::size_t i = 0; i < cl1.condition.size(); ++i)
            {
                double l1 = cl1.condition[i].l;
                double l2 = cl2.condition[i].l;
                cl1.condition[i].l = l1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (l2 - l1);
                cl2.condition[i].l = l1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (l2 - l1);

                double u1 = cl1.condition[i].u;
                double u2 = cl2.condition[i].u;
                cl1.condition[i].u = u1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (u2 - u1);
                cl2.condition[i].u = u1 + Random::nextDouble(-m_pConstants->blxAlpha, 1.0 + m_pConstants->blxAlpha) * (u2 - u1);
            }

            // Fix lower and upper order
            for (std::size_t i = 0; i < cl1.condition.size(); ++i)
            {
                if (cl1.condition[i].l > cl1.condition[i].u)
                {
                    std::swap(cl1.condition[i].l, cl1.condition[i].u);
                }

                if (cl2.condition[i].l > cl2.condition[i].u)
                {
                    std::swap(cl2.condition[i].l, cl2.condition[i].u);
                }
            }

            return true;
        }

        // APPLY MUTATION
        virtual void mutate(ClassifierType & cl, const std::vector<type> & situation) const override
        {
            assert(cl.condition.size() == situation.size());

            // Mutate lower or upper
            for (std::size_t i = 0; i < cl.condition.size(); ++i)
            {
                if (Random::nextDouble() < m_pConstants->mu)
                {
                    if (Random::nextDouble() < 0.5)
                    {
                        cl.condition[i].l += Random::nextDouble(-m_pConstants->mutationMaxChange, m_pConstants->mutationMaxChange);
                        if (m_pConstants->doRangeRestriction)
                        {
                            cl.condition[i].l = std::min(std::max(m_pConstants->minValue, cl.condition[i].l), m_pConstants->maxValue);
                        }
                    }
                    else
                    {
                        cl.condition[i].u += Random::nextDouble(-m_pConstants->mutationMaxChange, m_pConstants->mutationMaxChange);
                        if (m_pConstants->doRangeRestriction)
                        {
                            cl.condition[i].u = std::min(std::max(m_pConstants->minValue, cl.condition[i].u), m_pConstants->maxValue);
                        }
                    }
                }

                if (cl.condition[i].l > cl.condition[i].u)
                {
                    std::swap(cl.condition[i].l, cl.condition[i].u);
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
