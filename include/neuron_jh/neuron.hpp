#include<vector>

using value_t = float;

template<typename TActFn, typename TVal = value_t>
class Cynapse;

template<typename TActFn, typename TVal = value_t>
class Neuron
{
    using CynapseType = Cynapse<TActFn, TVal>;

    TVal value;
    TActFn activation_fn;
    
    std::vector<CynapseType*> inputs;
    std::vector<CynapseType*> outputs;
};

template<typename TActFn, typename TVal>
class Cynapse
{
    using NeuronType = Neuron<TActFn, TVal>;

    TVal weight;
    NeuronType* input;
    NeuronType* output;

};