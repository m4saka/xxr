#pragma once

#include "../xcs/ga.hpp"

namespace xxr { namespace xcsr_impl
{

    template <class Population>
    class GA : public xcs_impl::GA<Population>
    {
    public:
        using typename xcs_impl::GA<Population>::type;
        using typename xcs_impl::GA<Population>::SymbolType;
        using typename xcs_impl::GA<Population>::ConditionType;
        using typename xcs_impl::GA<Population>::ActionType;
        using typename xcs_impl::GA<Population>::ConditionActionPairType;
        using typename xcs_impl::GA<Population>::ConstantsType;
        using typename xcs_impl::GA<Population>::ClassifierType;
        using typename xcs_impl::GA<Population>::StoredClassifierType;
        using typename xcs_impl::GA<Population>::ClassifierPtr;
        using typename xcs_impl::GA<Population>::ClassifierPtrSetType;
        using typename xcs_impl::GA<Population>::PopulationType;

    protected:
        using xcs_impl::GA<Population>::m_constants;
        using xcs_impl::GA<Population>::m_availableActions;
        using xcs_impl::GA<Population>::uniformCrossover;
        using xcs_impl::GA<Population>::onePointCrossover;
        using xcs_impl::GA<Population>::twoPointCrossover;

        virtual bool blxAlphaCrossover(ClassifierType & cl1, ClassifierType & cl2) const = 0;

        // APPLY CROSSOVER
        virtual bool crossover(ClassifierType & cl1, ClassifierType & cl2) const override
        {
            switch (m_constants.crossoverMethod)
            {
            case ConstantsType::CrossoverMethod::UNIFORM_CROSSOVER:
                return uniformCrossover(cl1, cl2);

            case ConstantsType::CrossoverMethod::ONE_POINT_CROSSOVER:
                return onePointCrossover(cl1, cl2);

            case ConstantsType::CrossoverMethod::TWO_POINT_CROSSOVER:
                return twoPointCrossover(cl1, cl2);

            case ConstantsType::CrossoverMethod::BLX_ALPHA_CROSSOVER:
                return blxAlphaCrossover(cl1, cl2);
            
            default:
                return false;
            }
        }

    public:
        // Constructor
        using xcs_impl::GA<Population>::GA;

        // Destructor
        virtual ~GA() = default;
    };

}}
