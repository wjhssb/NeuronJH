#include <utility>

template<typename F>
struct RandomAccessWrapper
{
    F fn;

    template<typename Self>
    constexpr decltype(auto) operator[](this Self&& self, size_t i)
    {
        return std::forward_like<Self>(self.fn)(i);
    }

    struct iterator
    {
        RandomAccessWrapper& self;
        size_t i;

        
    };
};