#pragma once

#include <vector>
#include <set>
#include <memory>

namespace xxr { namespace xcs_impl
{

    template <class StoredClassifier>
    class ClassifierPtrSet
    {
    public:
        using type = typename StoredClassifier::type;
        using SymbolType = typename StoredClassifier::SymbolType;
        using ConditionType = typename StoredClassifier::ConditionType;
        using ActionType = typename StoredClassifier::ActionType;
        using ConditionActionPairType = typename StoredClassifier::ConditionActionPairType;
        using ConstantsType = typename StoredClassifier::ConstantsType;
        using ClassifierType = typename StoredClassifier::ClassifierType;
        using StoredClassifierType = StoredClassifier;
        using ClassifierPtr = std::shared_ptr<StoredClassifier>;

    protected:
        const ConstantsType * const m_pConstants;
        const std::unordered_set<ActionType> m_availableActions;

        std::set<ClassifierPtr> m_set;

    private:
        static std::set<ClassifierPtr> makeSetFromClassifiers(const std::vector<ClassifierType> & classifiers, const ConstantsType *pConstants)
        {
            std::set<ClassifierPtr> set;
            for (auto && cl : classifiers)
            {
                set.emplace(std::make_shared<StoredClassifier>(cl, pConstants));
            }
            return set;
        }

    public:
        // Constructor
        ClassifierPtrSet(const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : m_pConstants(pConstants)
            , m_availableActions(availableActions)
        {
        }

        ClassifierPtrSet(const std::set<ClassifierPtr> & set, const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : m_set(set)
            , m_pConstants(pConstants)
            , m_availableActions(availableActions)
        {
        }

        ClassifierPtrSet(const std::vector<ClassifierType> & initialClassifiers, const ConstantsType *pConstants, const std::unordered_set<ActionType> & availableActions)
            : m_set(makeSetFromClassifiers(initialClassifiers, pConstants))
            , m_pConstants(pConstants)
            , m_availableActions(availableActions)
        {
        }

        // Destructor
        virtual ~ClassifierPtrSet() = default;

        auto empty() const noexcept
        {
            return m_set.empty();
        }

        auto size() const noexcept
        {
            return m_set.size();
        }

        auto begin() const noexcept
        {
            return m_set.begin();
        }

        auto end() const noexcept
        {
            return m_set.end();
        }

        auto cbegin() const noexcept
        {
            return m_set.cbegin();
        }

        auto cend() const noexcept
        {
            return m_set.cend();
        }

        template <class... Args>
        auto insert(Args && ... args)
        {
            return m_set.insert(std::forward<Args>(args)...);
        }

        template <class... Args>
        auto emplace(Args && ... args)
        {
            return m_set.emplace(std::forward<Args>(args)...);
        }

        template <class... Args>
        auto erase(Args && ... args)
        {
            return m_set.erase(std::forward<Args>(args)...);
        }

        void clear() noexcept
        {
            m_set.clear();
        }

        template <class... Args>
        void swap(Args && ... args)
        {
            m_set.swap(std::forward<Args>(args)...);
        }

        template <class... Args>
        auto find(Args && ... args) const
        {
            return m_set.find(std::forward<Args>(args)...);
        }

        template <class... Args>
        auto count(Args && ... args) const
        {
            return m_set.count(std::forward<Args>(args)...);
        }
    };

}}
