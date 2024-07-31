#include <vector>
#include <cmath>//说了是cmath
#include <ranges>
#include <algorithm>

using value_t = float;

template<typename TActFn, typename TVal = value_t>
struct Cynapse;

template<typename TActFn, typename TVal = value_t>
class Neuron
{
    
    using CynapseType = Cynapse<TActFn, TVal>;
    friend  CynapseType;
    
    TVal value_;
    TActFn activation_fn_;
    
    std::vector<CynapseType*> inputs_;
    std::vector<CynapseType*> outputs_;

public:

    TVal& value()
    {
        return value_;
    }

    void update()
    {
        TVal input_sum = 0;
        for(auto input : inputs_)
        {
            input_sum += input->input->value_ * input->weight;
        }
        value_ = activation_fn_(input_sum);
    }

    friend CynapseType& connect(Neuron& input, Neuron& output, TVal weight)
    {
        CynapseType* cynapse = new CynapseType{ weight, &input, &output };
        input.outputs_.push_back(cynapse);
        output.inputs_.push_back(cynapse);
        return *cynapse;
    }

    ~Neuron()
    {
        //删除神经元会将与其连接的突出全部删除
        while(inputs_.size() != 0)
        {
            delete inputs_[0];
        }
        while(outputs_.size() != 0)
        {
            delete outputs_[0];
        }
    }
};

template<typename TActFn, typename TVal>
struct Cynapse
{
    using NeuronType = Neuron<TActFn, TVal>;

    TVal weight;
    NeuronType* input;
    NeuronType* output;

    ~Cynapse()//析构函数，如果这个突出被删除要做什么
    {
        //删除输入神经元里记录的自己
        std::erase_if(input->outputs_, [&](auto p){ return p == this; });
        //删除输出神经元里记录的自己
        std::erase_if(output->inputs_, [&](auto p){ return p == this; });
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