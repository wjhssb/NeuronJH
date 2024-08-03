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
};

template<typename TActFn, typename TVal>
class Neuron
{
    using AxonType = Axon<TActFn, TVal>;
    using DendriteType = Dendrite<TActFn, TVal>;
    friend DendriteType;

    TVal input_;//加了一个input避免重复运算
    TVal value_;
    TActFn activation_fn_;
    
    std::vector<DendriteType> dendrites_;
    std::vector<AxonType> axons_;

public:

    TVal& value()
    {
        return value_;
    }

    TVal input()const
    {
        return input_;
    }

    void update()
    {
        //正向传播的时候把input和value都设置好
        input_ = 0;
        for(auto [sender, axon_index] : dendrites_)
        {
            input_ += sender->axons_[axon_index].weight * sender->value();
        }
        value_ = activation_fn_(input_);
    }

    TVal inverse_input()const
    {
        TVal input = 0;
        for(auto [receiver, weight] : axons_)
        {
            input += weight * receiver->value();
        }
        return input;
    }

    //反向传播
    void inverse_update(TVal learning_rate)
    {
        //反向传播的值，暂存
        auto inverse_value = inverse_input() * derivation(activation_fn_)(input());

        for(auto& [receiver, weight] : axons_)
        {
            weight -= learning_rate * value_ * receiver->value();
        }

        value_ = inverse_value;
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

// template<typename TVal = value_t>
// struct SigmoidDer
// {
//     Sigmoid<TVal> sigmoid;

//     TVal operator()(TVal x)const
//     {
//         return sigmoid(x) * (static_cast<TVal>(1.0) - sigmoid(x));
//     }
// };

//这个就是取导数，没问题吧
template<typename TVal>
constexpr auto derivation(Sigmoid<TVal> sigmoid)
{
    //这样也行，一样的，反正sigmoid是空类
    //这个捕获了啥？
    return [=](TVal x){  return sigmoid(x) * (static_cast<TVal>(1.0) - sigmoid(x)); };
    // return [sigmoid](TVal x){  return sigmoid(x) * (static_cast<TVal>(1.0) - sigmoid(x)); };
    //return SigmoidDer<TVal>{ sigmoid };
    //这3句是一样的，能理解捕获啥意思了吧，就是lambda的类型里有个成员去存它的值就叫捕获它
}

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