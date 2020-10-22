#pragma once

#include <string>
#include <unordered_set>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <cassert>

namespace xxr { namespace xcs_impl
{

    template <class Condition, typename Action>
    struct ConditionActionPair
    {
    public:
        using type = typename Condition::type;
        using SymbolType = typename Condition::SymbolType;
        using ConditionType = Condition;
        using ActionType = Action;

        // C
        //   The condition specifies the input states (sensory situations)
        //   in which the classifier can be applied (matches).
        Condition condition;

        // A
        //   The action specifies the action (possibly a classification)
        //   that the classifier proposes.
        Action action;

        // Constructor
        ConditionActionPair(const ConditionActionPair &) = default;

        ConditionActionPair(const Condition & condition, Action action) : condition(condition), action(action) {}

        ConditionActionPair(Condition && condition, Action action) : condition(std::move(condition)), action(action) {}

        // Destructor
        virtual ~ConditionActionPair() = default;

        // IS MORE GENERAL
        virtual bool isMoreGeneral(const ConditionActionPair<Condition, Action> & cl) const
        {
            assert(condition.size() == cl.condition.size());

            bool ret = false;

            for (std::size_t i = 0; i < condition.size(); ++i)
            {
                if (condition.at(i) != cl.condition.at(i))
                {
                    if (!condition.at(i).isDontCare())
                    {
                        return false;
                    }
                    else 
                    {
                        ret = true;
                    }
                }
            }

            return ret;
        }

        friend std::ostream & operator<< (std::ostream & os, const ConditionActionPair<Condition, Action> & obj)
        {
            return os << obj.condition << ":" << obj.action;
        }
    };

    template <class ConditionActionPair>
    struct Classifier : ConditionActionPair
    {
    public:
        using typename ConditionActionPair::type;
        using typename ConditionActionPair::SymbolType;
        using typename ConditionActionPair::ConditionType;
        using typename ConditionActionPair::ActionType;
        using ConditionActionPairType = ConditionActionPair;

        // p
        //   The prediction p estimates (keeps an average of) the payoff expected if the
        //   classifier matches and its action is taken by the system.
        double prediction;

        // epsilon
        //   The prediction error epsilon estimates the errors made in the predictions.
        double epsilon;

        // F
        //   The fitness F denotes the classifier's fitness.
        double fitness;

        // exp
        //   The experience exp counts the number of times since its creation that the
        //   classifier has belonged to an action set.
        uint64_t experience;

        // ts
        //   The time stamp ts denotes the time-step of the last occurrence of a GA in
        //   an action set to which this classifier belonged.
        uint64_t timeStamp;

        // as
        //   The action set size as estimates the average size of the action sets this
        //   classifier has belonged to.
        double actionSetSize;

        // n
        //   The numerosity n reflects the number of micro-classifiers (ordinary
        //   classifiers) this classifier - which is technically called a macro-
        //   classifier - represents.
        uint64_t numerosity;

        using ConditionActionPair::action;
        using ConditionActionPair::isMoreGeneral;

        // Constructor
        Classifier(const Classifier &) = default;

        Classifier(const ConditionType & condition, ActionType action, double prediction, double epsilon, double fitness, uint64_t timeStamp)
            : ConditionActionPair(condition, action)
            , prediction(prediction)
            , epsilon(epsilon)
            , fitness(fitness)
            , experience(0)
            , timeStamp(timeStamp)
            , actionSetSize(1)
            , numerosity(1)
        {
        }

        Classifier(const ConditionActionPair & conditionActionPair, double prediction, double epsilon, double fitness, uint64_t timeStamp)
            : ConditionActionPair(conditionActionPair)
            , prediction(prediction)
            , epsilon(epsilon)
            , fitness(fitness)
            , experience(0)
            , timeStamp(timeStamp)
            , actionSetSize(1)
            , numerosity(1)
        {
        }

        Classifier(ConditionActionPair && conditionActionPair, double prediction, double epsilon, double fitness, uint64_t timeStamp)
            : ConditionActionPair(std::move(conditionActionPair))
            , prediction(prediction)
            , epsilon(epsilon)
            , fitness(fitness)
            , experience(0)
            , timeStamp(timeStamp)
            , actionSetSize(1)
            , numerosity(1)
        {
        }

        Classifier(const std::vector<type> & situation, ActionType action, double prediction, double epsilon, double fitness, uint64_t timeStamp)
            : Classifier(ConditionType(situation), action, prediction, epsilon, fitness, timeStamp)
        {
        }

        Classifier(const std::string & condition, ActionType action, double prediction, double epsilon, double fitness, uint64_t timeStamp)
            : Classifier(ConditionType(condition), action, prediction, epsilon, fitness, timeStamp)
        {
        }

        // Destructor
        virtual ~Classifier() = default;

        virtual double accuracy(double epsilonZero, double alpha, double nu) const
        {
            if (epsilon < epsilonZero)
            {
                return 1.0;
            }
            else
            {
                return alpha * pow(epsilon / epsilonZero, -nu);
            }
        }
    };

    // Classifier in [P] (have a reference to Constants)
    template <class Classifier, class Constants>
    struct StoredClassifier : Classifier
    {
    public:
        using typename Classifier::type;
        using typename Classifier::SymbolType;
        using typename Classifier::ConditionType;
        using typename Classifier::ActionType;
        using typename Classifier::ConditionActionPairType;
        using ClassifierType = Classifier;
        using ConstantsType = Constants;

        using Classifier::action;
        using Classifier::isMoreGeneral;
        using Classifier::prediction;
        using Classifier::epsilon;
        using Classifier::fitness;
        using Classifier::experience;
        using Classifier::timeStamp;
        using Classifier::actionSetSize;
        using Classifier::numerosity;

    protected:
        // Constants
        const Constants * const m_pConstants;

    public:
        // Constructor
        StoredClassifier(const StoredClassifier<Classifier, Constants> & obj) = default;

        StoredClassifier(const Classifier & obj, const Constants *pConstants)
            : Classifier(obj)
            , m_pConstants(pConstants)
        {
        }

        StoredClassifier(const ConditionType & condition, ActionType action, uint64_t timeStamp, const Constants *pConstants)
            : Classifier(condition, action, pConstants->initialPrediction, pConstants->initialEpsilon, pConstants->initialFitness, timeStamp)
            , m_pConstants(pConstants)
        {
        }

        StoredClassifier(const ConditionActionPairType & conditionActionPair, uint64_t timeStamp, const Constants *pConstants)
            : Classifier(conditionActionPair, pConstants->initialPrediction, pConstants->initialEpsilon, pConstants->initialFitness, timeStamp)
            , m_pConstants(pConstants)
        {
        }

        StoredClassifier(ConditionActionPairType && conditionActionPair, uint64_t timeStamp, const Constants *pConstants)
            : Classifier(std::move(conditionActionPair), pConstants->initialPrediction, pConstants->initialEpsilon, pConstants->initialFitness, timeStamp)
            , m_pConstants(pConstants)
        {
        }

        StoredClassifier(const std::vector<type> & situation, ActionType action, uint64_t timeStamp, const Constants *pConstants)
            : Classifier(situation, action, pConstants->initialPrediction, pConstants->initialEpsilon, pConstants->initialFitness, timeStamp)
            , m_pConstants(pConstants)
        {
        }

        StoredClassifier(const std::string & condition, ActionType action, uint64_t timeStamp, const Constants *pConstants)
            : Classifier(condition, action, pConstants->initialPrediction, pConstants->initialEpsilon, pConstants->initialFitness, timeStamp)
            , m_pConstants(pConstants)
        {
        }

        // Destructor
        virtual ~StoredClassifier() = default;

        // COULD SUBSUME
        virtual bool isSubsumer() const noexcept
        {
            return experience > m_pConstants->thetaSub && epsilon < m_pConstants->epsilonZero;
        }

        // DOES SUBSUME
        virtual bool subsumes(const Classifier & cl) const
        {
            return action == cl.action && isSubsumer() && isMoreGeneral(cl);
        }

        virtual double accuracy() const
        {
            return Classifier::accuracy(m_pConstants->epsilonZero, m_pConstants->alpha, m_pConstants->nu);
        }
    };

}}
