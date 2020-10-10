#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cassert>

namespace xxr { namespace xcs_impl
{

    template <class T>
    class AbstractSymbol
    {
    public:
        using type = T;

        virtual ~AbstractSymbol() = default;

        virtual std::string toString() const = 0;

        friend std::ostream & operator<< (std::ostream & os, const AbstractSymbol & obj)
        {
            return os << obj.toString();
        }

        virtual bool isDontCare() const = 0;

        virtual bool matches(T symbol) const = 0;

        virtual void setDontCare() = 0;
    };

    // The standard symbol for XCS (with "don't care")
    template <typename T>
    class Symbol : public AbstractSymbol<T>
    {
    protected:
        T m_value;
        bool m_isDontCare;

    public:
        // Constructor
        constexpr Symbol() : m_isDontCare(true) {}

        constexpr explicit Symbol(T value) : m_value(value), m_isDontCare(false) {}

        constexpr explicit Symbol(char c) : m_value(c - '0'), m_isDontCare(c == '#') {}

        constexpr Symbol(const Symbol<T> & obj) = default;

        // Destructor
        virtual ~Symbol() = default;

        virtual T value() const
        {
            assert(!m_isDontCare);
            return m_value;
        }

        virtual bool isDontCare() const override
        {
            return m_isDontCare;
        }

        virtual std::string toString() const override
        {
            if (isDontCare())
                return "# ";
            else
                return std::to_string(value()) + " ";
        }

        friend bool operator== (const Symbol<T> & lhs, const Symbol<T> & rhs)
        {
            return lhs.isDontCare() == rhs.isDontCare() && (lhs.isDontCare() || lhs.value() == rhs.value());
        }

        friend bool operator!= (const Symbol<T> & lhs, const Symbol<T> & rhs)
        {
            return !(lhs == rhs);
        }

        virtual Symbol<T> & operator= (const Symbol<T> & obj)
        {
            m_value = obj.m_value;
            m_isDontCare = obj.m_isDontCare;
            return *this;
        }

        // DOES MATCH
        virtual bool matches(T value) const noexcept override
        {
            return isDontCare() || this->value() == value;
        }

        virtual void setDontCare() override
        {
            m_isDontCare = true;
        }
    };

}}
