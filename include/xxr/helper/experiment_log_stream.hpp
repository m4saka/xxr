#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cstddef>
#include "simple_moving_average.hpp"

namespace xxr
{

    class ExperimentLogStream
    {
    private:
        std::ofstream m_ofs;

    protected:
        std::ostream & m_os;

    public:
        explicit ExperimentLogStream(const std::string & filename = "", bool useStdoutWhenEmpty = true)
            : m_os(
                filename.empty()
                    ? (useStdoutWhenEmpty ? std::cout : m_ofs)
                    : m_ofs)
        {
            m_ofs.open(filename);
        }

        virtual ~ExperimentLogStream() = default;

        void write(const std::string & str)
        {
            if (m_os)
            {
                m_os << str << std::flush;
            }
        }

        void writeLine(const std::string & str)
        {
            if (m_os)
            {
                m_os << str << std::endl;
            }
        }

        virtual void write(double value)
        {
            if (m_os)
            {
                m_os << value << std::flush;
            }
        }

        virtual void writeLine(double value)
        {
            if (m_os)
            {
                m_os << value << std::endl;
            }
        }
    };

    class SMAExperimentLogStream : public ExperimentLogStream
    {
    private:
        SimpleMovingAverage<double> m_sma;
        std::size_t m_count;

    public:
        explicit SMAExperimentLogStream(const std::string & filename = "", std::size_t smaWidth = 1, bool useStdoutWhenEmpty = true)
            : ExperimentLogStream(filename, useStdoutWhenEmpty)
            , m_sma(smaWidth)
        {
        }

        virtual void write(double value) override
        {
            if (m_os)
            {
                double smaValue = m_sma(value);
                if (++m_count >= m_sma.order())
                {
                    m_os << smaValue << std::flush;
                }
            }
        }

        virtual void writeLine(double value) override
        {
            if (m_os)
            {
                double smaValue = m_sma(value);
                if (++m_count >= m_sma.order())
                {
                    m_os << smaValue << std::endl;
                }
            }
        }
    };

}
