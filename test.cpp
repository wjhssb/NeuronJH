#include <iostream>
#include <neuron_jh/neuron.hpp>

int main()
{
    Neuron<Sigmoid<>> a{};
    Neuron<Sigmoid<>> b{};

    connect(a, b, 1.0f);

    a.set_value(0.5f);

    b.update();

    std::cout << b.get_value() << '\n';
    std::cout << Sigmoid<>{}(0.5f) << '\n';
}