#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <cstddef>

#include "repr.hpp"
#include "constants.hpp"

namespace xxr { namespace xcsr_impl
{

    // XCSR experiment class
    // (not a base class for XCSR experiments)
    template <
        typename T,
        typename Action
    >
    class Experiment : public AbstractExperiment<T, Action>
    {
    protected:
        std::unique_ptr<AbstractExperiment<T, Action>> m_experiment;
        const Repr m_repr;

    public:
        using type = T;
        using ActionType = Action;
        using ConstantsType = xxr::xcsr_impl::Constants;

        // Constructor
        explicit Experiment(const std::unordered_set<Action> & availableActions, const ConstantsType & constants, Repr repr)
            : m_repr(repr)
        {
            switch (repr)
            {
            case Repr::CSR:
                m_experiment = std::make_unique<csr::Experiment<T, Action>>(availableActions, constants);
                break;

            case Repr::OBR:
                m_experiment = std::make_unique<obr::Experiment<T, Action>>(availableActions, constants);
                break;

            case Repr::UBR:
                m_experiment = std::make_unique<ubr::Experiment<T, Action>>(availableActions, constants);
                break;

            default:
                assert(false);
            }
        }

        // Destructor
        virtual ~Experiment() = default;

        // Run with exploration
        virtual Action explore(const std::vector<T> & situation) override
        {
            return m_experiment->explore(situation);
        }

        virtual void reward(double value, bool isEndOfProblem = true) override
        {
            m_experiment->reward(value, isEndOfProblem);
        }

        // Run without exploration
        // (Set update to true when testing multi-step problems. If update is true, make sure to call reward() after this.)
        virtual Action exploit(const std::vector<T> & situation, bool update = false) override
        {
            return m_experiment->exploit(situation, update);
        }

        // Get prediction value of the previous action decision
        // (Call this function after explore() or exploit())
        virtual double prediction() const
        {
            return m_experiment->prediction();
        }

        virtual void loadPopulationCSV(const std::string & filename, bool useAsInitialPopulation = true) override
        {
            m_experiment->loadPopulationCSV(filename, useAsInitialPopulation);
        }

        template <class Classifier>
        void setPopulation(const std::vector<Classifier> & classifiers, bool initTimeStamp = true)
        {
            m_experiment->setPopulation(classifiers, initTimeStamp);
        }

        virtual void dumpPopulation(std::ostream & os) const override
        {
            m_experiment->dumpPopulation(os);
        }

        virtual std::size_t populationSize() const noexcept override
        {
            return m_experiment->populationSize();
        }

        virtual std::size_t numerositySum() const override
        {
            return m_experiment->numerositySum();
        }

        virtual void switchToCondensationMode() noexcept override
        {
            m_experiment->switchToCondensationMode();
        }

        virtual Repr repr() noexcept
        {
            return m_repr;
        }
    };

}}
