#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <cassert>
#include <limits>

#include "environment.hpp"
#include "../random.hpp"

namespace xxr
{

    class FunctionEnvironment final : public AbstractEnvironment<double, int>
    {
    private:
        std::function<double(const std::vector<double> &)> m_func;
        const std::size_t m_dim;
        std::vector<double> m_situation;
        bool m_isEndOfProblem;

        static std::vector<double> randomSituation(std::size_t dim)
        {
            std::vector<double> situation;
            for (std::size_t i = 0; i < dim; ++i)
            {
                situation.push_back(Random::nextDouble());
            }
            return situation;
        }

    public:
        explicit FunctionEnvironment(std::function<double(const std::vector<double> &)> func, std::size_t dim)
            : AbstractEnvironment<double, int>({ 0 })
            , m_func(func)
            , m_dim(dim)
            , m_situation(randomSituation(dim))
            , m_isEndOfProblem(false)
        {
        }

        ~FunctionEnvironment() = default;

        virtual std::vector<double> situation() const override
        {
            return m_situation;
        }

        virtual double executeAction(int action) override
        {
            double reward = m_func(m_situation);

            // Update situation
            m_situation = randomSituation(m_dim);

            // Single-step problem
            m_isEndOfProblem = true;

            return reward;
        }

        virtual bool isEndOfProblem() const override
        {
            return m_isEndOfProblem;
        }

        // Returns the payoff for the given situation
        double getRewardAnswer(const std::vector<double> & situation) const
        {
            assert(situation.size() == m_dim);

            return m_func(m_situation);
        }
    };

}