#pragma once

#include <cassert>

#include "../XCS/environment.h"
#include "../XCS/random.h"
#include "symbol.h"

namespace XCSR
{

    class RealMultiplexerEnvironment final : public XCS::AbstractEnvironment<double, bool, Symbol<double>>
    {
    private:
        const std::size_t m_totalLength;
        const std::size_t m_addressBitLength;
        const std::size_t m_registerBitLength;
        const double m_binaryThreshold;
        std::vector<double> m_situation;
        bool m_isEndOfProblem;

        // Get address bit length from total length
        static constexpr std::size_t addressBitLength(std::size_t l, std::size_t c)
        {
            return (l == 0) ? c - 1 : addressBitLength(l >> 1, c + 1);
        }

        static std::vector<double> randomSituation(std::size_t totalLength)
        {
            std::vector<double> situation;
            for (std::size_t i = 0; i < totalLength; ++i)
            {
                situation.push_back(XCS::Random::nextDouble());
            }
            return situation;
        }

    public:
        explicit RealMultiplexerEnvironment(std::size_t length, double binaryThreshold = 0.5) :
            AbstractEnvironment<double, bool, Symbol<double>>({ false, true }),
            m_totalLength(length),
            m_addressBitLength(addressBitLength(length, 0)),
            m_registerBitLength(length - m_addressBitLength),
            m_binaryThreshold(binaryThreshold),
            m_situation(randomSituation(length)),
            m_isEndOfProblem(false)
        {
            // Total length must be n + 2^n (n > 0)
            assert(m_totalLength == (m_addressBitLength + ((std::size_t)1 << m_addressBitLength)));
        }

        ~RealMultiplexerEnvironment() = default;

        virtual std::vector<double> situation() const override
        {
            return m_situation;
        }

        virtual double executeAction(bool action) override
        {
            double reward = (action == getAnswer(m_situation)) ? 1.0 : 0.0;

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

        // Returns the answer
        virtual bool getAnswer(const std::vector<double> & situation) const
        {
            std::size_t address = 0;
            for (std::size_t i = 0; i < m_addressBitLength; ++i)
            {
                if (situation.at(i) >= m_binaryThreshold)
                {
                    address += (std::size_t)1 << (m_addressBitLength - i - 1);
                }
            }

            return situation.at(m_addressBitLength + address) >= m_binaryThreshold;
        }
    };

}