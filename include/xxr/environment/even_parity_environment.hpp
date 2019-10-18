#pragma once

#include <vector>
#include <cstddef>
#include <cassert>

#include "environment.hpp"
#include "../random.hpp"

namespace xxr
{

    class EvenParityEnvironment final : public AbstractEnvironment<bool, bool>
    {
    private:
        const std::size_t m_length;
        std::vector<bool> m_situation;
        bool m_isEndOfProblem;

        static std::vector<bool> randomSituation(std::size_t length)
        {
            std::vector<bool> situation;
            situation.reserve(length);
            for (std::size_t i = 0; i < length; ++i)
            {
                situation.push_back(Random::nextInt(0, 1));
            }
            return situation;
        }

    public:
        // Constructor
        explicit EvenParityEnvironment(std::size_t length)
            : AbstractEnvironment<bool, bool>({ false, true })
            , m_length(length)
            , m_situation(randomSituation(length))
            , m_isEndOfProblem(false)
        {
        }

        virtual ~EvenParityEnvironment() = default;

        virtual std::vector<bool> situation() const override
        {
            return m_situation;
        }

        virtual double executeAction(bool action) override
        {
            double reward = (action == getAnswer()) ? 1000.0 : 0.0;

            // Update situation
            m_situation = randomSituation(m_length);

            // Single-step problem
            m_isEndOfProblem = true;

            return reward;
        }

        virtual bool isEndOfProblem() const override
        {
            return m_isEndOfProblem;
        }

        // Returns answer to situation
        bool getAnswer() const
        {
            int sum = 0;
            for (bool b : m_situation)
            {
                sum += static_cast<int>(b);
            }
            return (sum % 2) == 0; // even => 1, odd => 0
        }
    };

}
