#pragma once

#include <random>
#include <vector>
#include <set>
#include <unordered_set>
#include <limits>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <iostream>

namespace xxr
{

    class Random
    {
    private:
        static auto && device()
        {
            static std::random_device device;
            return device;
        }

        static auto && engine()
        {
            static std::mt19937 engine(device()());
            return engine;
        }

    public:
        template <typename T = double>
        static T nextDouble(T min = 0.0, T max = 1.0)
        {
            std::uniform_real_distribution<T> dist(min, max);
            return dist(engine());
        }

        template <typename T = int>
        static T nextInt(T min, T max)
        {
            std::uniform_int_distribution<T> dist(min, max);
            return dist(engine());
        }

        template <typename T>
        static auto chooseFrom(const std::vector<T> & container)
        {
            auto size = container.size();

            assert(size > 0);

            std::uniform_int_distribution<decltype(size)> dist(0, size - 1);
            return *(std::begin(container) + dist(engine()));
        }

        template <typename T>
        static auto chooseFrom(const std::set<T> & container)
        {
            std::vector<T> vec;
            vec.reserve(container.size());
            vec.insert(vec.end(), container.begin(), container.end());
            return chooseFrom(vec);
        }

        template <typename T>
        static auto chooseFrom(const std::unordered_set<T> & container)
        {
            std::vector<T> vec;
            vec.reserve(container.size());
            vec.insert(vec.end(), container.begin(), container.end());
            return chooseFrom(vec);
        }

        template <typename T>
        static std::size_t rouletteWheelSelection(const std::vector<T> & container)
        {
            // Prepare a roulette wheel by the weights
            T sum = 0;
            std::vector<T> rouletteWheel;
            rouletteWheel.reserve(container.size());
            for (auto && value : container)
            {
                sum += value;
                rouletteWheel.push_back(sum);
            }

            assert(sum > 0);

            // Spin the roulette wheel
            auto it = std::lower_bound(std::begin(rouletteWheel), std::end(rouletteWheel), nextDouble<T>(0, sum));

            // Returns index of selected item
            return std::distance(std::begin(rouletteWheel), it);
        }

        template <typename T>
        static std::size_t greedySelection(const std::vector<T> & container)
        {
            T maxValue = *std::max_element(std::begin(container), std::end(container));
            std::vector<std::size_t> maxValueIdxs;

            for (std::size_t i = 0; i < container.size(); ++i)
            {
                if (container[i] == maxValue)
                {
                    maxValueIdxs.push_back(i);
                }
            }

            assert(maxValueIdxs.size() > 0);

            return chooseFrom(maxValueIdxs);
        }

        template <typename T>
        static std::size_t epsilonGreedySelection(const std::vector<T> & container, double epsilon)
        {
            std::size_t selectedIdx;

            if (nextDouble() < epsilon)
            {
                // Random selection
                std::size_t size = container.size();
                assert(size > 0);
                selectedIdx = nextInt<std::size_t>(0, size - 1);
            }
            else
            {
                // Greedy selection
                selectedIdx = greedySelection(container);
            }

            return selectedIdx;
        }

        template <typename T>
        static std::size_t tournamentSelection(const std::vector<T> & container, double tau)
        {
            std::size_t selectedIdx = container.size() - 1;
            T best = std::numeric_limits<T>::lowest();

            for (std::size_t i = 0; i < container.size(); ++i)
            {
                if (nextDouble() < tau && best < container[i])
                {
                    best = container[i];
                    selectedIdx = i;
                }
            }

            if (best == std::numeric_limits<T>::lowest())
            {
                return nextInt<std::size_t>(0, container.size() - 1);
            }
            else
            {
                return selectedIdx;
            }
        }

        template <typename T>
        static std::size_t tournamentSelectionMicroClassifier(const std::vector<std::pair<T, std::size_t>> & container, double tau)
        {
            std::size_t selectedIdx = container.size() - 1;
            T best = std::numeric_limits<T>::lowest();

            for (std::size_t i = 0; i < container.size(); ++i)
            {
                if (best < container[i].first / container[i].second)
                {
                    for (std::size_t j = 0; j < container[i].second /*numerosity*/; ++j)
                    {
                        if (nextDouble() < tau)
                        {
                            best = container[i].first / container[i].second;
                            selectedIdx = i;
                            break;
                        }
                    }
                }
            }

            if (best == std::numeric_limits<T>::lowest())
            {
                return nextInt<std::size_t>(0, container.size() - 1);
            }
            else
            {
                return selectedIdx;
            }
        }
    };

}
