#include <concepts>
#include <iostream>
#include <neuron_jh/neuron.hpp>


struct Lambda0
{
    int offset;

    auto operator()(int x)const
    {
        return x + offset;
    }
};

int fn(int x)
{
    return x;
}

int fn2(int x)
{
    return x + 233;
}

int main()
{
    Neuron<Sigmoid<>> a{};
    Neuron<Sigmoid<>> b{};

    Neuron<Sigmoid<>>::connect(a, b, 1.0f);

    a.value() = 0.5f;

    b.update();

    std::cout << b.value() << '\n';
    std::cout << Sigmoid<>{}(0.5f) << '\n';
}