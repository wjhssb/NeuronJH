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
    
    TVal value;
    TActFn activation_fn;
    
    std::vector<CynapseType*> inputs;
    std::vector<CynapseType*> outputs;

public:

    void set_value(TVal value)
    {
        this->value = value;
    }

    TVal get_value()const
    {
        return value;
    }

    void update()
    {
        TVal input_sum = 0;
        for(auto input : inputs)
        {
            input_sum += input->input->value * input->weight;
        }
        value = activation_fn(input_sum);
    }

    friend CynapseType& connect(Neuron& input, Neuron& output, TVal weight)
    {
        CynapseType* cynapse = new CynapseType{ weight, &input, &output };
        input.outputs.push_back(cynapse);
        output.inputs.push_back(cynapse);
        return *cynapse;
    }

    ~Neuron()
    {
        //删除神经元会将与其连接的突出全部删除
        for(auto input : inputs)
        {
            delete input;
        }
        for(auto output : outputs)
        {
            delete output;
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
        std::ranges::remove_if(input->outputs, [&](auto p){ return p == this; });
        //删除输出神经元里记录的自己
        std::ranges::remove_if(output->inputs, [&](auto p){ return p == this; });
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