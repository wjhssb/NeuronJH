#include <cstddef>
#include <vector>
#include <cmath>//说了是cmath
#include <ranges>
#include <algorithm>

using value_t = float;

template<typename TActFn, typename TVal = value_t>
class Neuron;

//轴突
template<typename TActFn, typename TVal = value_t>
struct Axon
{
    using NeuronType = Neuron<TActFn, TVal>;
    NeuronType* receiver;
    TVal weight;
};

//树突
template<typename TActFn, typename TVal = value_t>
struct Dendrite
{
    using NeuronType = Neuron<TActFn, TVal>;
    NeuronType* sender;
    size_t axon_index;

    TVal value()const
    {
        return sender->axons_[axon_index].weight * sender->value();
    }
};

template<typename TActFn, typename TVal>
class Neuron
{
    using AxonType = Axon<TActFn, TVal>;
    using DendriteType = Dendrite<TActFn, TVal>;
    friend DendriteType;

    TVal value_;
    TActFn activation_fn_;
    
    std::vector<DendriteType> dendrites_;
    std::vector<AxonType> axons_;

public:

    TVal& value()
    {
        return value_;
    }

    void update()
    {
        TVal input_sum = 0;
        for(auto dendrite : dendrites_)
        {
            input_sum += dendrite.value();
        }
        value_ = activation_fn_(input_sum);
    }

    static void connect(Neuron& sender, Neuron& receiver, TVal weight)
    {
        sender.axons_.push_back({ &receiver, weight });
        receiver.dendrites_.push_back({ &sender, sender.axons_.size() - 1 });
    }
};

template<typename TVal = value_t>
struct Sigmoid
{
    TVal operator()(TVal x)const
    {
        return static_cast<TVal>(1.0) / (static_cast<TVal>(1.0) + std::exp(-x));
    }
};

template<typename TVal = value_t>
struct Tanh
{
    TVal operator()(TVal x)const
    {
        return std::tanh(x);
    }
};

template<typename TVal = value_t>
struct ReLU//?
{
    TVal operator()(TVal x)const
    {
        return std::max(static_cast<TVal>(0.0), x);
    }
};

template<typename TVal = value_t>
struct LeakyReLU
{
    TVal operator()(TVal x)const
    {
        if( x < static_cast<TVal>(0.0))
        {
            return static_cast<TVal>(0.25) * (std::exp(x) - static_cast<TVal>(1.0));
        }
        else
        {
            return x;
        }
    }
};