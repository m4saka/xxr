#pragma once
#include <cstddef>
#include <cassert>

namespace xxr
{

    template <typename T>
    class UnrecursiveFilter
    {
    protected:
        const std::size_t m_order; // How many samples to use
        std::size_t m_cursor;
        std::size_t m_valueCount;

        // Ring buffer (Use dynamic allocation to use the heap memory in case of a large order)
        T *m_pBuffer;

        explicit UnrecursiveFilter(std::size_t order) : m_order(order), m_cursor(0), m_valueCount(0)
        {
            // Filter order must not be zero
            assert(m_order > 0);

            m_pBuffer = new T[m_order];
        }

        ~UnrecursiveFilter()
        {
            delete[] m_pBuffer;
        }

        // Make sure to call this first in operator()
        void storeValue(T value)
        {
            // Store the given value to the buffer
            m_pBuffer[m_cursor] = value;

            // Increment the value count
            if (m_valueCount < m_order)
                ++m_valueCount;

            // Move the cursor to the next
            if (++m_cursor >= m_order)
                m_cursor = 0;
        }

    public:
        UnrecursiveFilter(const UnrecursiveFilter &) = delete;
        UnrecursiveFilter & operator=(const UnrecursiveFilter &) = delete;

        virtual T operator()(T value) = 0;

        std::size_t order() const
        {
            return m_order;
        }
    };

    // Simple Moving Average
    template <typename T>
    class SimpleMovingAverage : public UnrecursiveFilter<T>
    {
    private:
        using UnrecursiveFilter<T>::m_valueCount;
        using UnrecursiveFilter<T>::m_pBuffer;

    public:
        explicit SimpleMovingAverage(std::size_t order) : UnrecursiveFilter<T>(order) {}

        T operator()(T value) override
        {
            this->storeValue(value);

            // Total sum
            T sum = m_pBuffer[0];
            for (std::size_t i = 1; i < m_valueCount; ++i)
            {
                sum += m_pBuffer[i];
            }

            // Return the average
            return sum / m_valueCount;
        }

        using UnrecursiveFilter<T>::order;
    };

}
