#pragma once

#include <vector>
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
        std::vector<bool> m_situation;
        bool m_isEndOfProblem;

        // Get address bit length from total length
        static constexpr std::size_t addressBitLength(std::size_t l, std::size_t c)
        {
            return (l == 0) ? c - 1 : addressBitLength(l >> 1, c + 1);
        }

        static std::vector<bool> randomSituation(std::size_t totalLength)
        {
            std::vector<bool> situation;
            situation.reserve(totalLength);
            for (std::size_t i = 0; i < totalLength; ++i)
            {
                situation.push_back(Random::nextInt(0, 1));
            }
            return situation;
        }

    public:
        // Constructor
        explicit MultiplexerEnvironment(std::size_t length)
            : AbstractEnvironment<bool, bool>({ false, true })
            , m_totalLength(length)
            , m_addressBitLength(addressBitLength(length, 0))
            , m_situation(randomSituation(length))
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
            m_situation = randomSituation(m_totalLength);

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
            std::size_t address = 0;
            for (std::size_t i = 0; i < m_addressBitLength; ++i)
            {
                if (m_situation.at(i) == true)
                {
                    address += (std::size_t)1 << (m_addressBitLength - i - 1);
                }
            }

            return m_situation.at(m_addressBitLength + address) == true;
        }
    };

}
