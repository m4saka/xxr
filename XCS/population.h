#pragma once

#include "symbol.h"
#include "classifier_ptr_set.h"

namespace XCS
{

    template <typename T, typename Action, class Symbol, class Condition, class Classifier>
    class Population : public ClassifierPtrSet<Action, Classifier>
    {
    protected:
        using ClassifierPtr = std::shared_ptr<Classifier>;
        using ClassifierPtrSet<Action, Classifier>::m_set;
        using ClassifierPtrSet<Action, Classifier>::m_constants;
        using ClassifierPtrSet<Action, Classifier>::m_actionChoices;

        // DELETION VOTE
        virtual double deletionVote(const Classifier & cl, double averageFitness) const
        {
            double vote = cl.actionSetSize * cl.numerosity;

            // Consider fitness for deletion vote
            if ((cl.experience > m_constants.thetaDel) && (cl.fitness / cl.numerosity < averageFitness))
            {
                vote *= averageFitness / (cl.fitness / cl.numerosity);
            }

            return vote;
        }

    public:
        using ClassifierPtrSet<Action, Classifier>::ClassifierPtrSet;

        // INSERT IN POPULATION
        virtual void insertOrIncrementNumerosity(const Classifier & cl)
        {
            for (auto && c : m_set)
            {
                if (static_cast<ConditionActionPair<T, Action, Symbol, Condition>>(*c).equals(cl))
                {
                    ++c->numerosity;
                    return;
                }
            }
            m_set.insert(std::make_shared<Classifier>(cl));
        }

        // DELETE FROM POPULATION
        virtual void deleteExtraClassifiers()
        {
            uint64_t numerositySum = 0;
            double fitnessSum = 0.0;
            for (auto && c : m_set)
            {
                numerositySum += c->numerosity;
                fitnessSum += c->fitness;
            }

            // Return if the sum of numerosity has not met its maximum limit
            if (numerositySum < m_constants.maxPopulationClassifierCount)
            {
                return;
            }

            // The average fitness in the population
            double averageFitness = fitnessSum / numerositySum;

            // Prepare a roulette wheel by the weights
            double voteSum = 0.0;
            std::vector<double> rouletteWheel(m_set.size());
            std::vector<const ClassifierPtr *> rouletteWheelTarget(m_set.size());
            for (auto && c : m_set)
            {
                voteSum += deletionVote(*c, averageFitness);
                rouletteWheel.push_back(voteSum);
                rouletteWheelTarget.push_back(&c);
            }

            // Spin the roulette wheel
            auto rouletteIterator = std::lower_bound(std::begin(rouletteWheel), std::end(rouletteWheel), Random::nextDouble(0.0, voteSum));
            auto rouletteIdx = std::distance(std::begin(rouletteWheel), rouletteIterator);

            // Distrust the selected classifier
            if ((*rouletteWheelTarget[rouletteIdx])->numerosity > 1)
            {
                (*rouletteWheelTarget[rouletteIdx])->numerosity--;
            }
            else
            {
                m_set.erase(*rouletteWheelTarget[rouletteIdx]);
            }
        }
    };

}