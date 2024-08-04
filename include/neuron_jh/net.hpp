#include <cstddef>
#include <tuple>
#include <vector>

namespace neuronjh 
{
    //网络中每一层的类型可以不同啊
    template<typename...TLayer>
    class Net
    {
        using ValueType = decltype((..., std::declval<TLayer>()))::ValueType;
        //元组，用来存一组类型不同的值
        std::tuple<TLayer...> layers_;
    public:
        template<typename...L>
        constexpr Net(L&&...layers) : layers_{ std::forward<L>(layers)... }
        {}

        //ok了
        //整个网络的输入是input时，第I层的输出（默认最后一层）
        template<size_t I = sizeof...(TLayer) - 1>
        auto forward(auto input)
        {
            if constexpr(I == 0)//这个I == 0是编译期常量，且两个分支返回值不一定是同一个类型，所以必须if constexpr，这相当于编译期直接判断走哪个分支，另一个分支直接不存在了
            {
                return std::get<0>(layers_).forward(input);
            }
            else
            {
                return std::get<I>(layers_).forward(forward<I - 1>(input));
            }
        }

        template<size_t I = sizeof...(TLayer) - 1>
        auto get_field()const
        {
            struct NodeField
            {
                ValueType output;
                ValueType gradient;
            };

            std::array<std::vector<NodeField>, sizeof...(TLayer)> field;
        }

    private:
        template<size_t I = 0>
        auto init_field()const
        {
            if constexpr()
        }
    };
}