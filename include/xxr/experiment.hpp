#pragma once
#include <vector>
#include <iostream>
#include <cstddef>

namespace xxr {

    template <
        typename T,
        typename Action
    >
    class AbstractExperiment
    {
    public:
        // Destructor
        virtual ~AbstractExperiment() = default;

        // Run with exploration
        virtual Action explore(const std::vector<T> & situation) = 0;
        virtual Action explore(const std::vector<T> & situation, const std::vector<T> & situationSigma)
        {
            std::cout << "AbstractExperiment::explore(const std::vector<T> &, const std::vector<T> &) is not implemented!" << std::endl;
            std::exit(1);
        }

        virtual void reward(double value, bool isEndOfProblem = true) = 0;

        // Run without exploration
        // (Set update to true when testing multi-step problems. If update is true, make sure to call reward() after this.)
        virtual Action exploit(const std::vector<T> & situation, bool update = false) = 0;
        virtual Action exploit(const std::vector<T> & situation, const std::vector<T> & situationSigma, bool update = false)
        {
            std::cout << "AbstractExperiment::exploit(const std::vector<T> &, const std::vector<T> &, bool) is not implemented!" << std::endl;
            std::exit(1);
        }

        virtual double prediction() const = 0;

        virtual double predictionFor(int action) const = 0;

        virtual bool isCoveringPerformed() const = 0;

        virtual void loadPopulationCSV(const std::string & filename, bool useAsInitialPopulation = true) = 0;

        virtual void dumpPopulation(std::ostream & os) const = 0;

        virtual std::size_t populationSize() const = 0;

        virtual std::size_t numerositySum() const = 0;

        virtual void switchToCondensationMode() noexcept = 0;
    };

}
