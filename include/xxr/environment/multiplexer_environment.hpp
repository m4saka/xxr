#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <cassert>

#include "environment.hpp"
#include "../random.hpp"

namespace xxr
{

    class MultiplexerEnvironment final : public AbstractEnvironment<bool, bool>
    {
    private:
        const std::size_t m_totalLength;
        const std::size_t m_addressBitLength;
        const double m_minorityAcceptanceProbability;
        std::vector<bool> m_situation;
        bool m_isEndOfProblem;

        // Get address bit length from total length
        static constexpr std::size_t addressBitLength(std::size_t l, std::size_t c)
        {
            return (l == 0) ? c - 1 : addressBitLength(l >> 1, c + 1);
        }

        static bool getAnswerOfSituation(const std::vector<bool> & situation)
        {
            std::size_t address = 0;
            auto l = addressBitLength(situation.size(), 0);
            for (std::size_t i = 0; i < l; ++i)
            {
                if (situation.at(i) == true)
                {
                    address += (std::size_t)1 << (l - i - 1);
                }
            }

            return situation.at(l + address) == true;
        }

        static std::vector<bool> randomSituation(std::size_t totalLength, double minorityAcceptanceProbability = 1.0)
        {
            std::vector<bool> situation;
            while (true)
            {
                for (std::size_t i = 0; i < totalLength; ++i)
                {
                    situation.push_back(Random::nextInt(0, 1));
                }

                if (getAnswerOfSituation(situation) == 1)
                {
                    break;
                }
                else if (Random::nextDouble() < minorityAcceptanceProbability)
                {
                    break;
                }
            }
            return situation;
        }

    public:
        // Constructor
        // (Set imbalanceLevel to higher than 0 for imbalanced multiplexer problems)
        explicit MultiplexerEnvironment(std::size_t length, unsigned int imbalanceLevel = 0)
            : AbstractEnvironment<bool, bool>({ false, true })
            , m_totalLength(length)
            , m_addressBitLength(addressBitLength(length, 0))
            , m_minorityAcceptanceProbability(1.0 / std::pow(2, imbalanceLevel))
            , m_situation(randomSituation(length, m_minorityAcceptanceProbability))
            , m_isEndOfProblem(false)
        {
            // Total length must be n + 2^n (n > 0)
            assert(m_totalLength == (m_addressBitLength + ((std::size_t)1 << m_addressBitLength)));
        }

        virtual ~MultiplexerEnvironment() = default;

        virtual std::vector<bool> situation() const override
        {
            return m_situation;
        }

        virtual double executeAction(bool action) override
        {
            double reward = (action == getAnswer()) ? 1000.0 : 0.0;

            // Update situation
            m_situation = randomSituation(m_totalLength, m_minorityAcceptanceProbability);

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
            return getAnswerOfSituation(m_situation);
        }
    };

}
