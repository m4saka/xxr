#pragma once
#include <cmath>

#include "../../xcs/match_set.hpp"

namespace xxr { namespace xcsr_impl { namespace csr
{

    template <class Population>
    class MatchSet : public xcs_impl::MatchSet<Population>
    {
    public:
        using typename xcs_impl::MatchSet<Population>::type;
        using typename xcs_impl::MatchSet<Population>::SymbolType;
        using typename xcs_impl::MatchSet<Population>::ConditionType;
        using typename xcs_impl::MatchSet<Population>::ActionType;
        using typename xcs_impl::MatchSet<Population>::ConditionActionPairType;
        using typename xcs_impl::MatchSet<Population>::ConstantsType;
        using typename xcs_impl::MatchSet<Population>::ClassifierType;
        using typename xcs_impl::MatchSet<Population>::StoredClassifierType;
        using typename xcs_impl::MatchSet<Population>::ClassifierPtr;
        using typename xcs_impl::MatchSet<Population>::ClassifierPtrSetType;
        using typename xcs_impl::MatchSet<Population>::PopulationType;

    protected:
        using xcs_impl::MatchSet<Population>::m_set;
        using xcs_impl::MatchSet<Population>::m_pConstants;
        using xcs_impl::MatchSet<Population>::m_availableActions;
        using xcs_impl::MatchSet<Population>::m_isCoveringPerformed;

        // GENERATE COVERING CLASSIFIER
        virtual ClassifierPtr generateCoveringClassifier(const std::vector<type> & situation, const std::unordered_set<ActionType> & unselectedActions, uint64_t timeStamp) const override
        {
            std::cout << "Error: Deleted function xcsr_impl::MatchSet::exploit is called!" << std::endl;
            return nullptr;
        }
        virtual ClassifierPtr generateCoveringClassifier(const std::vector<type> & situation, const std::vector<type> & situationSigma, const std::unordered_set<ActionType> & unselectedActions, uint64_t timeStamp) const
        {
            assert(situation.size() == situationSigma.size());

            std::vector<SymbolType> symbols;
            for (std::size_t i = 0; i < situation.size(); ++i)
            {
                symbols.emplace_back(situation[i], std::abs(situationSigma[i]) * 0.5/*FIXME*/);
            }

            return std::make_shared<StoredClassifierType>(symbols, Random::chooseFrom(unselectedActions), timeStamp, m_pConstants);
        }

    public:
        // Constructor
        MatchSet(const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : xcs_impl::MatchSet<Population>(pConstants, availableActions)
        {
        }

        MatchSet(Population & population, const std::vector<type> & situation, uint64_t timeStamp, const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : MatchSet(pConstants, availableActions)
        {
            std::cout << "Error: Deleted function xcsr_impl::MatchSet::MatchSet is called!" << std::endl;
            std::exit(1);
        }

        MatchSet(Population & population, const std::vector<type> & situation, const std::vector<type> & situationSigma, uint64_t timeStamp, const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : MatchSet(pConstants, availableActions)
        {
            this->regenerate(population, situation, situationSigma, timeStamp);
        }

        // GENERATE MATCH SET
        virtual void regenerate(Population & population, const std::vector<type> & situation, uint64_t timeStamp) override
        {
            std::cout << "Error: Deleted function xcsr_impl::MatchSet::regenerate is called!" << std::endl;
            std::exit(1);
        }
        virtual void regenerate(Population & population, const std::vector<type> & situation, const std::vector<type> & situationSigma, uint64_t timeStamp)
        {
            // Set theta_mna (the minimal number of actions) to the number of action choices if theta_mna is 0
            auto thetaMna = (m_pConstants->thetaMna == 0) ? m_availableActions.size() : m_pConstants->thetaMna;

            auto unselectedActions = m_availableActions;

            m_set.clear();

            while (m_set.empty())
            {
                for (auto && cl : population)
                {
                    if (cl->condition.matches(situation))
                    {
                        m_set.insert(cl);
                        unselectedActions.erase(cl->action);
                    }
                }

                // Generate classifiers covering the unselected actions
                if (m_availableActions.size() - unselectedActions.size() < thetaMna)
                {
                    auto coveringClassifier = generateCoveringClassifier(situation, situationSigma/* <- This is the only change here */, unselectedActions, timeStamp);
                    if (!coveringClassifier->condition.matches(situation))
                    {
                        std::cerr <<
                            "Error: The covering classifier does not contain the current situation!\n"
                            "  - Current situation: ";
                        for (auto && s : situation)
                        {
                            std::cerr << s << " ";
                        }
                        std::cerr << "\n  - Covering classifier: " << *coveringClassifier << "\n" << std::endl;
                        assert(false);
                    }
                    population.insert(coveringClassifier);
                    population.deleteExtraClassifiers();
                    m_set.clear();
                    m_isCoveringPerformed = true;
                }
                else
                {
                    m_isCoveringPerformed = false;
                }
            }
        }

        // Destructor
        virtual ~MatchSet() = default;
    };

}}}
