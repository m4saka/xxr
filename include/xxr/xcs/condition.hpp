#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstddef>
#include <cassert>
#include <iterator>

#include "../random.hpp"

namespace xxr { namespace xcs_impl
{

    template <class Symbol>
    class Condition
    {
    public:
        using type = typename Symbol::type;
        using SymbolType = Symbol;

    protected:
        std::vector<Symbol> m_symbols;

    public:
        // Constructor
        Condition() = default;

        Condition(const std::vector<Symbol> & symbols) : m_symbols(symbols) {}

        Condition(const std::vector<type> & symbols) : m_symbols(symbols.begin(), symbols.end()) {}

        explicit Condition(const std::string & symbols)
        {
            std::istringstream iss(symbols);
            std::string symbol;
            while (std::getline(iss, symbol, ' '))
            {
                if (symbol == "#")
                {
                    m_symbols.emplace_back('#');
                }
                else
                {
                    m_symbols.emplace_back(std::stoi(symbol));
                }
            }
        }

        // Destructor
        virtual ~Condition() = default;

        virtual std::string toString() const
        {
            std::string str;
            for (auto && symbol : m_symbols)
            {
                str += symbol.toString();
            }

            // Erase last whitespace
            if (!str.empty() && str.back() == ' ')
            {
                str.pop_back();
            }

            return str;
        }

        virtual Symbol & operator[] (std::size_t idx)
        {
            return m_symbols[idx];
        }

        virtual const Symbol & at(std::size_t idx) const
        {
            return m_symbols.at(idx);
        }

        friend std::ostream & operator<< (std::ostream & os, const Condition & obj)
        {
            return os << obj.toString();
        }

        friend bool operator== (const Condition & lhs, const Condition & rhs)
        {
            return lhs.m_symbols == rhs.m_symbols;
        }

        friend bool operator!= (const Condition & lhs, const Condition & rhs)
        {
            return lhs.m_symbols != rhs.m_symbols;
        }

        // DOES MATCH
        virtual bool matches(const std::vector<type> & situation) const
        {
            assert(m_symbols.size() == situation.size());

            for (std::size_t i = 0; i < m_symbols.size(); ++i)
            {
                if (!m_symbols[i].matches(situation[i]))
                {
                    return false;
                }
            }

            return true;
        }

        auto empty() const noexcept
        {
            return m_symbols.empty();
        }

        auto size() const noexcept
        {
            return m_symbols.size();
        }

        auto begin() const noexcept
        {
            return m_symbols.begin();
        }

        auto end() const noexcept
        {
            return m_symbols.end();
        }

        auto cbegin() const noexcept
        {
            return m_symbols.cbegin();
        }

        auto cend() const noexcept
        {
            return m_symbols.cend();
        }

        virtual void setDontCareAtRandom(double dontCareProbability)
        {
            for (auto && symbol : m_symbols)
            {
                if (Random::nextDouble() < dontCareProbability)
                {
                    symbol.setDontCare();
                }
            }
        }

        virtual std::size_t dontCareCount() const
        {
            std::size_t count = 0;
            for (auto && symbol : m_symbols)
            {
                if (symbol.isDontCare())
                {
                    count++;
                }
            }

            return count;
        }
    };

}}
