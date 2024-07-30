#include<vector>

class Cynapse;

template<typename TActFn, typename TVal = float>
class Neuron
{
    TVal value;
    std::vector<Cynapse*> input;
    std::vector<Cynapse*> output;
    TActFn activation_fn;
};