#pragma once
#include <istream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstddef>

#include "dataset.hpp"
#include "xxr/xcs/classifier.hpp"

namespace xxr
{

    namespace CSV
    {
        template <typename T>
        void saveCSV(std::ostream & os, const std::vector<std::vector<T>> & data)
        {
            for (const auto & line : data)
            {
                for (std::size_t i = 0; i < line.size(); ++i)
                {
                    os << line[i];
                    if (i != line.size() - 1)
                    {
                        os << ',';
                    }
                }
                os << '\n';
            }
        }

        template <typename T>
        std::vector<std::vector<T>> readSituations(std::istream & is, bool rounds = false)
        {
            std::vector<std::vector<T>> situations;

            // Load all lines from csv
            std::string line;
            while (std::getline(is, line) && !line.empty())
            {
                // Split comma-separated string
                std::istringstream iss(line);
                std::string field;
                double fieldValue;
                std::vector<T> situation;
                bool empty = true;
                while (std::getline(iss, field, ','))
                {
                    fieldValue = std::stof(field);
                    situation.push_back(static_cast<T>(rounds ? std::round(fieldValue) : fieldValue));
                    empty = false;
                }

                if (empty)
                {
                    break; // this gets rid of maybe-uninitialized warning
                }

                situations.push_back(situation);
            }

            return situations;
        }

        template <typename T, typename Action>
        Dataset<T, Action> readDataset(std::istream & is, bool rounds = false)
        {
            std::vector<std::vector<T>> situations;
            std::vector<Action> actions;

            // Load all lines from csv
            std::string line;
            while (std::getline(is, line) && !line.empty())
            {
                // Split comma-separated string
                std::istringstream iss(line);
                std::string field;
                double fieldValue;
                std::vector<T> situation;
                bool empty = true;
                while (std::getline(iss, field, ','))
                {
                    fieldValue = std::stof(field);
                    situation.push_back(static_cast<T>(rounds ? std::round(fieldValue) : fieldValue));
                    empty = false;
                }

                if (empty)
                {
                    break; // this gets rid of maybe-uninitialized warning
                }

                // Last field is action
                actions.push_back(static_cast<Action>(fieldValue));
                situation.pop_back();

                situations.push_back(situation);
            }

            return { situations, actions };
        }

        template <typename T, typename Action>
        Dataset<T, Action> readDataset(const std::string & filename, bool rounds = false)
        {
            std::ifstream ifs(filename);
            if (!ifs.good())
            {
                throw new std::runtime_error("Error: Cannot open file '" + filename + "'");
            }
            return readDataset<T, Action>(ifs, rounds);
        }

        template <class Classifier>
        std::vector<Classifier> readPopulation(std::istream & is, bool skipFirstLine = true, bool skipFirstColumn = false)
        {
            using T = typename Classifier::type;
            using Action = typename Classifier::ActionType;

            std::vector<Classifier> classifiers;

            std::string line;

            if (skipFirstLine)
            {
                std::getline(is, line);
            }

            // Load all lines from csv
            while (std::getline(is, line) && !line.empty())
            {
                // Split comma-separated string
                std::istringstream iss(line);
                std::string field;
                std::vector<T> situation;
                bool empty = true;

                // For XCSR population CSV, we need an option whether to skip the first graphic notation column
                if (skipFirstColumn)
                {
                    std::getline(iss, field, ',');
                }

                using Condition = typename Classifier::ConditionType;
                Condition condition;
                Action action;
                double prediction;
                double epsilon;
                double fitness;
                std::size_t experience;
                std::size_t timeStamp;
                double actionSetSize;
                std::size_t numerosity;

                int columnIdx = 0;
                while (std::getline(iss, field, ','))
                {
                    if (columnIdx == 0) // Condition
                    {
                        condition = Condition(field);
                    }
                    else
                    {
                        std::istringstream fieldIss(field);
                        switch (columnIdx)
                        {
                        case 1: // Action
                            fieldIss >> action;
                            break;
                        case 2: // Prediction
                            fieldIss >> prediction;
                            break;
                        case 3: // Epsilon
                            fieldIss >> epsilon;
                            break;
                        case 4: // Fitness
                            fieldIss >> fitness;
                            break;
                        case 5: // Experience
                            fieldIss >> experience;
                            break;
                        case 6: // TimeStamp
                            fieldIss >> timeStamp;
                            break;
                        case 7: // ActionSetSize
                            fieldIss >> actionSetSize;
                            break;
                        case 8: // Numerosity
                            fieldIss >> numerosity;
                            break;
                        }
                    }
                    ++columnIdx;
                    empty = false;
                }

                if (empty)
                {
                    break; // this gets rid of maybe-uninitialized warning
                }

                // Add classifier to list
                Classifier cl(condition, action, prediction, epsilon, fitness, timeStamp);
                cl.experience = experience;
                cl.actionSetSize = actionSetSize;
                cl.numerosity = numerosity;
                classifiers.push_back(std::move(cl));
            }

            return classifiers;
        }

        template <class Classifier>
        std::vector<Classifier> readPopulation(const std::string & filename, bool skipFirstLine = true, bool skipFirstColumn = false)
        {
            std::ifstream ifs(filename);
            if (!ifs.good())
            {
                throw new std::runtime_error("Error: Cannot open file '" + filename + "'");
            }
            return readPopulation<Classifier>(ifs, skipFirstLine, skipFirstColumn);
        }
    }

}
