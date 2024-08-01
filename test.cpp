#include <concepts>
#include <iostream>
#include <neuron_jh/neuron.hpp>


struct Lambda0
{
    int offset;

    auto operator()(int x)const
    {
        return x + offset;
    }
};

int fn(int x)
{
    return x;
}

int fn2(int x)
{
    return x + 233;
}

int main()
{
    Neuron<Sigmoid<>> a{};

    Neuron<Sigmoid<>> b1{};
    Neuron<Sigmoid<>> b2{};
    Neuron<Sigmoid<>> b3{};
    Neuron<Sigmoid<>> b4{};

    Neuron<Sigmoid<>> c{};

    Neuron<Sigmoid<>>::connect(a, b1, 1.0f);
    Neuron<Sigmoid<>>::connect(a, b2, 0.5f);
    Neuron<Sigmoid<>>::connect(a, b3, 0.3f);
    Neuron<Sigmoid<>>::connect(a, b4, -0.7f);
    Neuron<Sigmoid<>>::connect(b1, c, -1.0f);
    Neuron<Sigmoid<>>::connect(b2, c, 1.0f);
    Neuron<Sigmoid<>>::connect(b3, c, -1.0f);
    Neuron<Sigmoid<>>::connect(b4, c, 1.0f);

    float learning_rate = 0.5f;

    

    float max_loss = 0.0f;
    size_t n = 0;
    do
    {
        ++n;
        max_loss = 0.0f;
        for(float x = 0.0f; x < 1.1f; x += 0.1f)
        {
            a.value() = x;
            b1.update();
            b2.update();
            b3.update();
            b4.update();
            c.update();

            float target = 0.5f * x;
            float error = c.value() - target;
            float loss = error * error;

            c.value() = 2.0f * error * derivation(Sigmoid<>{})(c.input());
            b1.inverse_update(learning_rate);
            b2.inverse_update(learning_rate);
            b3.inverse_update(learning_rate);
            b4.inverse_update(learning_rate);
            a.inverse_update(learning_rate);

            if(loss > max_loss)
            {
                max_loss = loss;
            }
        }
        //std::cout << "max_loss: " << max_loss << std::endl;
        //system("PAUSE");
    }while(max_loss > 0.00001f);
    
    std::cout << "learning complete. n:" << n << " max_loss: " << max_loss << std::endl;

    float x;
    while((std::cout << "input: ", std::cin >> x))
    {
        a.value() = x;
        b1.update();
        b2.update();
        b3.update();
        b4.update();
        c.update();

        float target = 0.5f * x;
        float error = c.value() - target;
        float loss = error * error;

        std::cout << ", output: " << c.value() << ", target: " << target << ", loss: " << loss << std::endl;
    }
}