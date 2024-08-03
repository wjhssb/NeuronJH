#include <cstddef>
#include <vector>
#include <cmath>
#include <ranges>
#include <algorithm>

template<typename TVal, typename TActFn>
class FullyConnectedLayer
{

    size_t input_size_;
    size_t output_size_;

    //你这个Tval不就很有可能是float
    std::vector<TVal> weights_;
    TActFn activation_fn_;

public:
    FullyConnectedLayer(size_t input_size, size_t output_size, TActFn activation_fn)
     : input_size_{ input_size }
     , output_size_{ output_size }
     , weights_(input_size * output_size)//这个不能用花括号初始化
     , activation_fn_{ activation_fn }
    {
        //其实这里有个问题，就是weight怎么初始化值，先不管，你需要随机的初始weight啊
        //每一层单独开随机数引擎填充是不对的，因为很容易导致每一层被填充同样的序列。
        //由于权重的值可以任意的修改而不破坏层对象，所以可以把weight暴露出去，让外部统一随机初始化它们的值。
        //但是注意不能直接暴露weights_, 因为weights_的大小不能被随意修改。
    }

//能用下标取元素，且元素类型能和Tval做乘法结果也是Tval类型；满足前述条件的类型都可以作为input
    auto forward(auto input)const
    {
        std::vector<TVal> output{};
        output.resize(output_size_);

        for(size_t i = 0; i < output_size_; ++i)
        {
            output[i] = 0;
            for(size_t j = 0; j < input_size_; ++j)
            {
                output += weights_[input_size_ * i + j] * input[j];
            }
            output[i] = activation_fn_(output[i]);
        }
        
        return output;
    }

    //这样就暴露所有weight但是不给修改长度了，返回的是引用，所以可以修改
    TVal& weight(size_t i_input, size_t i_output)
    {
        return weights_[i_input + i_output * input_size_];
    }
};