#include <concepts>
#include <iostream>
#include <neuron_jh/neuron.hpp>


void unary_fn_test(auto fn, float start, float end, float step = 0.05f, float loss_target = 0.00001f, float learning_rate = 0.5f)
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

    float max_loss = 0.0f;
    size_t n = 0;
    do
    {
        ++n;
        max_loss = 0.0f;
        for(float x = start; x < end; x += step)
        {
            a.value() = x;
            b1.update();
            b2.update();
            b3.update();
            b4.update();
            c.update();

            float target = fn(x);
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
    }while(max_loss > loss_target);
    
    std::cout << "learning complete. learning cycles count:" << n << " final_max_loss: " << max_loss << std::endl;

    float x;
    char buff[255]{};
    
    while((std::cout << "input: ", std::cin.getline(buff, 254), strlen(buff) != 0))
    {
        x = atof(buff);

        a.value() = x;
        b1.update();
        b2.update();
        b3.update();
        b4.update();
        c.update();

        float target = fn(x);
        float error = c.value() - target;
        float loss = error * error;

        std::cout << "output: " << c.value() << ", target: " << target << ", loss: " << loss << std::endl;
    }

    std::cout << "unary_fn_test exit.\n";
}

int main()
{
    unary_fn_test([](auto x){ return 0.5f * x * x; }, 0.0f, 1.1f, 0.1f, 0.00001f, 0.5f);
}