#include <random>

namespace neuronjh 
{
    using value_t = float;

    template<typename TVal = value_t>
    inline auto& default_random_generator()
    {
        static std::default_random_engine random_engine{ 233 };
        static std::uniform_real_distribution<TVal> uniform_dist{ -1.0f, 1.0f };
        static auto generator = [&](){ return uniform_dist(random_engine); };
        return generator;
    };
}