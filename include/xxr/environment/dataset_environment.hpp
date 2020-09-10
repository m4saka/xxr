#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <cstddef>
#include <cassert>

#include "environment.hpp"
#include "xxr/random.hpp"
#include "xxr/helper/dataset.hpp"

namespace xxr
{

    template <typename T, typename Action>
    class DatasetEnvironment : public AbstractEnvironment<T, Action>
    {
    protected:
        const Dataset<T, Action> m_dataset;
        std::vector<T> m_situation;
        Action m_answer;
        std::size_t m_nextIdx;
        const bool m_chooseRandom;
        bool m_isEndOfProblem;

        virtual std::size_t loadNext()
        {
            if (m_chooseRandom)
            {
                const auto idx = Random::nextInt<std::size_t>(0UL, m_dataset.situations.size() - 1UL);
                m_situation = m_dataset.situations[idx];
                m_answer = m_dataset.actions[idx];
                return idx;
            }
            else
            {
                m_situation = m_dataset.situations[m_nextIdx];
                m_answer = m_dataset.actions[m_nextIdx];
                const auto idx = m_nextIdx;
                if (++m_nextIdx >= m_dataset.situations.size())
                {
                    m_nextIdx = 0;
                }
                return idx;
            }
        }

    public:
        DatasetEnvironment(const Dataset<T, Action> & dataset, const std::unordered_set<Action> & availableActions, bool chooseRandom = true)
            : AbstractEnvironment<T, Action>(availableActions)
            , m_dataset(dataset)
            , m_nextIdx(0)
            , m_chooseRandom(chooseRandom)
            , m_isEndOfProblem(false)
        {
            assert(!m_dataset.situations.empty());
            assert(!m_dataset.actions.empty());
            assert(m_dataset.situations.size() == m_dataset.actions.size());

            loadNext();
        }

        virtual ~DatasetEnvironment() = default;

        virtual std::vector<T> situation() const override
        {
            return m_situation;
        }

        virtual double executeAction(Action action) override
        {
            const double reward = (action == m_answer) ? 1000.0 : 0.0;

            // Single-step problem
            m_isEndOfProblem = true;

            loadNext();

            return reward;
        }

        virtual bool isEndOfProblem() const override
        {
            return m_isEndOfProblem;
        }

        // Returns the answer
        virtual Action getAnswer() const
        {
            return m_answer;
        }
    };

}
