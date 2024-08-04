#include <cstddef>
#include <vector>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <span>

#include "random.hpp"

namespace neuronjh 
{

    template<typename TVal, typename TActFn>
    class FullyConnectedLayer
    {
    public:
        using ValueType = TVal;
        using ActivationFnType = TActFn;
        using GradientFnType = decltype(gradient(std::declval<const TActFn&>()));

    private:
        size_t            input_size_;
        size_t            output_size_;
        std::vector<TVal> weights_;
        ActivationFnType  activation_fn_;
        GradientFnType    gradient_fn_;
    
    public:
        constexpr FullyConnectedLayer(
            size_t input_size, 
            size_t output_size, 
            TActFn activation_fn, 
            auto&& random_generator = default_random_generator()
        )
         : input_size_{ input_size }
         , output_size_{ output_size }
         , weights_(input_size * output_size)//这个不能用花括号初始化
         , activation_fn_{ activation_fn }
         , gradient_fn_{ gradient(activation_fn) }
        {
            for(auto& weight : weights_)
            {
                weight = random_generator();
            }
        }
    
        constexpr size_t input_size()const
        {
            return input_size_;
        }
    
        constexpr size_t output_size()const
        {
            return output_size_;
        }
    
        struct OutputType
        {
        const FullyConnectedLayer& layer;
        std::span<const TVal> input;

        //这个类的对象相当于layer输入input的结果向量，
        //但是结果向量的每个分量只有在通过下面的下标运算符取出时才会真正进行计算
        TVal operator[](size_t i)const
        {
            TVal input_sum{};
            for(size_t j = 0; j < layer.input_size_; ++j)
            {
                input_sum += layer.weights_[layer.input_size_ * i + j] * input[j];
            }
            return layer.activation_fn_(input_sum);
        }
        };
    
        //           std::span相当于两个指针，首指针指向连续元素序列的首个元素，尾指针指向最后一个元素的末尾
        //           它可以从任意的连续容器被构造出来（c数组，array，vector），相当于连续容器都可以往这里填
        constexpr OutputType forward(std::span<const TVal> input)const
        {
        //此处没有进行任何分配操作，相当于只是简单将自身引用和输入的span绑在一起返回
        return OutputType{ *this, input };

        //std::vector<TVal> output{};
        //output.resize(output_size_);

        // for(size_t i = 0; i < output_size_; ++i)
        // {
        //     output[i] = 0;
        //     for(size_t j = 0; j < input_size_; ++j)
        //     {
        //         output += weights_[input_size_ * i + j] * input[j];
        //     }
        //     output[i] = activation_fn_(output[i]);
        // }
        
        //return output;
        }

        struct OutputWithGradient
        {
            const FullyConnectedLayer& layer;
            std::span<const TVal> input;
    
            auto operator[](size_t i)const
            {
                TVal input_sum{};
                for(size_t j = 0; j < layer.input_size_; ++j)
                {
                    input_sum += layer.weights_[layer.input_size_ * i + j] * input[j];
                }
                struct ResultType
                {
                    TVal output;
                    TVal gradient;
                };
                return ResultType{ layer.activation_fn_(input_sum), layer.gradient_fn_(input_sum) };
            }
        };
    
        constexpr OutputWithGradient forward_with_gradient(std::span<TVal> input)const
        {
            return OutputWithGradient{ *this, input };
        }
    
        //这样就暴露所有weight但是不给修改长度了，返回的是引用，所以可以修改
        constexpr TVal& weight(size_t i_input, size_t i_output)
        {
            return weights_[i_input + i_output * input_size_];
        }
    };

}