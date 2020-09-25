#pragma once
#include <vector>

namespace xxr
{

    template <typename T, typename Action>
    struct Dataset
    {
        std::vector<std::vector<T>> situations;
        std::vector<std::vector<T>> situationSigmas;
        std::vector<Action> actions;
    };

    // Normalize data by min/max value
    template <typename T>
    void normalize(std::vector<std::vector<T>> & situations, T min, T max)
    {
        assert(min != max);

        for (auto && situation : situations)
        {
            for (auto && value : situation)
            {
                value = (value - min) / (max - min);
            }
        }
    }

    template <typename T>
    void normalize(std::vector<std::vector<T>> & situations, const std::pair<T, T> & minmax)
    {
        normalize(situations, minmax.first, minmax.second);
    }

    // Normalize data between 0.0 to 1.0 and return min/max value
    template <typename T>
    std::pair<T, T> normalize(std::vector<std::vector<T>> & situations)
    {
        if (situations.empty())
        {
            return { 0.0, 1.0 };
        }

        T min = situations[0][0];
        T max = situations[0][0];
        for (auto && situation : situations)
        {
            for (auto && value : situation)
            {
                if (min > value) min = value;
                if (max < value) max = value;
            }
        }

        normalize(situations, min, max);

        return { min, max };
    }

    template <typename T>
    void normalizeScale(std::vector<std::vector<T>> & situationSigmas, T min, T max)
    {
        if (min != 0.0 || max != 1.0)
        {
            for (auto && situationSigma : situationSigmas)
            {
                for (auto && value : situationSigma)
                {
                    value = value / (max - min);
                }
            }
        }
    }

    // Denormalize data by min/max value
    template <typename T>
    void denormalize(std::vector<std::vector<T>> & situations, T min, T max)
    {
        if (min != 0.0 || max != 1.0)
        {
            for (auto && situation : situations)
            {
                for (auto && value : situation)
                {
                    value = value * (max - min) + min;
                }
            }
        }
    }

    template <typename T>
    void denormalize(std::vector<std::vector<T>> & situations, const std::pair<T, T> & minmax)
    {
        denormalize(minmax.first, minmax.second);
    }

    template <typename T>
    void denormalizeScale(std::vector<std::vector<T>> & situationSigmas, T min, T max)
    {
        if (min != 0.0 || max != 1.0)
        {
            for (auto && situationSigma : situationSigmas)
            {
                for (auto && value : situationSigma)
                {
                    value = value * (max - min);
                }
            }
        }
    }

}
