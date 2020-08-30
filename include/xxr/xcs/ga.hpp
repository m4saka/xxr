#pragma once

#include <memory>
#include <vector>
#include <unordered_set>
#include <cassert>
#include <cstddef>

namespace xxr { namespace xcs_impl
{

    template <class Population>
    class GA
    {
    public:
        using type = typename Population::type;
        using SymbolType = typename Population::SymbolType;
        using ConditionType = typename Population::ConditionType;
        using ActionType = typename Population::ActionType;
        using ConditionActionPairType = typename Population::ConditionActionPairType;
        using ConstantsType = typename Population::ConstantsType;
        using ClassifierType = typename Population::ClassifierType;
        using StoredClassifierType = typename Population::StoredClassifierType;
        using ClassifierPtr = typename Population::ClassifierPtr;
        using ClassifierPtrSetType = typename Population::ClassifierPtrSetType;
        using PopulationType = Population;

    protected:
        const ConstantsType * const m_pConstants;
        const std::unordered_set<ActionType> & m_availableActions;

        // SELECT OFFSPRING
        virtual ClassifierPtr selectOffspring(const ClassifierPtrSetType & actionSet) const
        {
            std::vector<const ClassifierPtr *> targets;
            for (auto && cl : actionSet)
            {
                targets.push_back(&cl);
            }

            std::size_t selectedIdx;
            if (m_pConstants->tau > 0.0 && m_pConstants->tau <= 1.0)
            {
                // Tournament selection
                std::vector<std::pair<double, std::size_t>> fitnesses;
                fitnesses.reserve(actionSet.size());
                for (auto && target : targets)
                {
                    fitnesses.emplace_back((*target)->fitness, (*target)->numerosity);
                }
                selectedIdx = Random::tournamentSelectionMicroClassifier(fitnesses, m_pConstants->tau);
            }
            else
            {
                // Roulette-wheel selection
                std::vector<double> fitnesses;
                fitnesses.reserve(actionSet.size());
                for (auto && target : targets)
                {
                    fitnesses.push_back((*target)->fitness);
                }
                selectedIdx = Random::rouletteWheelSelection(fitnesses);
            }
            return *targets[selectedIdx];
        }

        // APPLY CROSSOVER (uniform crossover)
        virtual bool uniformCrossover(ClassifierType & cl1, ClassifierType & cl2) const
        {
            assert(cl1.condition.size() == cl2.condition.size());

            bool isChanged = false;
            for (std::size_t i = 0; i < cl1.condition.size(); ++i)
            {
                if (Random::nextDouble() < 0.5)
                {
                    std::swap(cl1.condition[i], cl2.condition[i]);
                    isChanged = true;
                }
            }
            return isChanged;
        }

        // APPLY CROSSOVER (one point crossover)
        virtual bool onePointCrossover(ClassifierType & cl1, ClassifierType & cl2) const
        {
            assert(cl1.condition.size() == cl2.condition.size());

            std::size_t x = Random::nextInt<std::size_t>(0, cl1.condition.size());

            bool isChanged = false;
            for (std::size_t i = x + 1; i < cl1.condition.size(); ++i)
            {
                std::swap(cl1.condition[i], cl2.condition[i]);
                isChanged = true;
            }
            return isChanged;
        }

        // APPLY CROSSOVER (two point crossover)
        virtual bool twoPointCrossover(ClassifierType & cl1, ClassifierType & cl2) const
        {
            assert(cl1.condition.size() == cl2.condition.size());

            std::size_t x = Random::nextInt<std::size_t>(0, cl1.condition.size());
            std::size_t y = Random::nextInt<std::size_t>(0, cl1.condition.size());

            if (x > y)
            {
                std::swap(x, y);
            }

            bool isChanged = false;
            for (std::size_t i = x + 1; i < y; ++i)
            {
                std::swap(cl1.condition[i], cl2.condition[i]);
                isChanged = true;
            }
            return isChanged;
        }

        // APPLY CROSSOVER
        virtual bool crossover(ClassifierType & cl1, ClassifierType & cl2) const
        {
            switch (m_pConstants->crossoverMethod)
            {
            case ConstantsType::CrossoverMethod::UNIFORM_CROSSOVER:
                return uniformCrossover(cl1, cl2);

            case ConstantsType::CrossoverMethod::ONE_POINT_CROSSOVER:
                return onePointCrossover(cl1, cl2);

            case ConstantsType::CrossoverMethod::TWO_POINT_CROSSOVER:
                return twoPointCrossover(cl1, cl2);
            
            default:
                return false;
            }
        }

        // APPLY MUTATION
        virtual void mutate(ClassifierType & cl, const std::vector<type> & situation) const
        {
            assert(cl.condition.size() == situation.size());

            for (std::size_t i = 0; i < cl.condition.size(); ++i)
            {
                if (Random::nextDouble() < m_pConstants->mu)
                {
                    if (cl.condition[i].isDontCare())
                    {
                        cl.condition[i] = SymbolType(situation.at(i));
                    }
                    else
                    {
                        cl.condition[i].setDontCare();
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

        void insertDiscoveredClassifiers(const ClassifierType & child1, const ClassifierType & child2, const ClassifierPtr & parent1, const ClassifierPtr & parent2, PopulationType & population) const
        {
            if (m_pConstants->doGASubsumption)
            {
                subsumeClassifier(child1, parent1, parent2, population);
                subsumeClassifier(child2, parent1, parent2, population);
            }
            else
            {
                population.insertOrIncrementNumerosity(std::make_shared<StoredClassifierType>(child1, m_pConstants));
                population.insertOrIncrementNumerosity(std::make_shared<StoredClassifierType>(child2, m_pConstants));
            }

            while (population.deleteExtraClassifiers()) {}
        }

        void subsumeClassifier(const ClassifierType & child, const ClassifierPtr & parent1, const ClassifierPtr & parent2, PopulationType & population) const
        {
            if (parent1->subsumes(child))
            {
                ++parent1->numerosity;
            }
            else if (parent2->subsumes(child))
            {
                ++parent2->numerosity;
            }
            else
            {
                subsumeClassifier(child, population); // calls second subsumeClassifier function!
            }
        }

        void subsumeClassifier(const ClassifierType & child, PopulationType & population) const
        {
            std::vector<ClassifierPtr> choices;

            for (auto && cl : population)
            {
                if (cl->subsumes(child))
                {
                    choices.push_back(cl);
                }
            }

            if (!choices.empty())
            {
                std::size_t choice = Random::nextInt<std::size_t>(0, choices.size() - 1);
                ++choices[choice]->numerosity;
                return;
            }

            population.insertOrIncrementNumerosity(std::make_shared<StoredClassifierType>(child, m_pConstants));
        }

    public:
        // Constructor
        GA(const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : m_pConstants(pConstants)
            , m_availableActions(availableActions)
        {
        }

        // Destructor
        virtual ~GA() = default;

        // RUN GA (refer to ActionSet::runGA() for the former part)
        virtual void run(ClassifierPtrSetType & actionSet, const std::vector<type> & situation, PopulationType & population) const
        {
            auto parent1 = selectOffspring(actionSet);
            auto parent2 = selectOffspring(actionSet);

            assert(parent1->condition.size() == parent2->condition.size());

            ClassifierType child1(*parent1);
            ClassifierType child2(*parent2);
            child1.fitness = parent1->fitness / parent1->numerosity;
            child2.fitness = parent2->fitness / parent2->numerosity;
            child1.numerosity = child2.numerosity = 1;
            child1.experience = child2.experience = 0;

            bool isChangedByCrossover;
            if (Random::nextDouble() < m_pConstants->chi)
            {
                isChangedByCrossover = crossover(child1, child2);
            }
            else
            {
                isChangedByCrossover = false;
            }

            mutate(child1, situation);
            mutate(child2, situation);

            if (isChangedByCrossover)
            {
                child1.prediction =
                    child2.prediction = (child1.prediction + child2.prediction) / 2;

                child1.epsilon =
                    child2.epsilon = (child1.epsilon + child2.epsilon) / 2;

                child1.fitness =
                    child2.fitness = (child1.fitness + child2.fitness) / 2 * 0.1; // fitnessReduction
            }
            else
            {
                child1.fitness *= 0.1; // fitnessReduction
                child2.fitness *= 0.1; // fitnessReduction
            }

            insertDiscoveredClassifiers(child1, child2, parent1, parent2, population);
        }
    };

}}
