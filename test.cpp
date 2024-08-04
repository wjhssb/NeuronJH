#include <concepts>
#include <iostream>
#include <random>
#include <array>
#include <fstream>
#include <functional>
#include <thread>
#include <chrono>
#include <semaphore>
#include <ranges>
#include <neuron_jh/neuron.hpp>

#include <neuron_jh/fully_connected_layer.hpp>
//#include <neuron_jh/neuron.hpp>

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
            float y = fn(x);
            a.value() = x;
           
            b1.update();
            b2.update();
            b3.update();
            b4.update();
            c.update();

            float target = y;
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

auto read_minist_set(const char* label_path, const char* image_path)
{
    std::ifstream label_file{ label_path, std::ios::in | std::ios::binary };
    label_file >> std::noskipws;
    label_file.ignore(4);
    int32_t label_count;
    label_file.read(reinterpret_cast<char *>(&label_count), sizeof(label_count));
    label_count = std::byteswap(label_count);
    //std::cout << "label count:" << label_count << "\n";

    std::vector<unsigned char> labels;
    labels.resize(static_cast<size_t>(label_count));
    label_file.read(reinterpret_cast<char *>(labels.data()), label_count);
    label_file.close();

    std::ifstream image_file{ image_path, std::ios::in | std::ios::binary };
    image_file >> std::noskipws;
    image_file.ignore(4);
    int32_t image_count;
    int32_t image_height;
    int32_t image_width;
    image_file.read(reinterpret_cast<char *>(&image_count), sizeof(image_count));
    image_count = std::byteswap(image_count);
    image_file.read(reinterpret_cast<char *>(&image_height), sizeof(image_height));
    image_height = std::byteswap(image_height);
    image_file.read(reinterpret_cast<char *>(&image_width), sizeof(image_width));
    image_width = std::byteswap(image_width);

    if(label_count != image_count)
    {
        std::puts("label count not equal to image count!");
    }

    size_t image_size = image_height * image_width;
    size_t pixel_count = image_count * image_size;

    std::vector<unsigned char> images;
    images.resize(pixel_count);
    image_file.read(reinterpret_cast<char *>(images.data()), pixel_count);
    
    image_file.close();

    struct ResultType
    {
        size_t count;
        size_t image_size;
        std::vector<unsigned char> labels;
        std::vector<float> images;
    };

    std::cout << "read " << labels.size() << " data.\n";

    return ResultType{ 
        labels.size(),
        image_size, std::move(labels), 
        images | std::views::transform([](auto x){ return x / 255.0f; }) | std::ranges::to<std::vector>()
    };
}

class thread
{
    std::function<void()> task_;
    std::binary_semaphore have_task_{ 0 };
    std::binary_semaphore task_complete_{ 0 };
    std::thread base_;

public:
    thread() : base_{ [&]{ while(true){ have_task_.acquire(); task_(); task_complete_.release(); } } }
    {}

    template<typename F>
    void do_task(F&& task)
    {
        task_ = std::forward<F>(task);
        have_task_.release();
    }

    void wait_idle()
    {
        task_complete_.acquire();
    }
};

void for_each(size_t size, auto&& fn)
{
    constexpr size_t n = 4;
    static thread threads[n - 1]{};

    size_t block_size = size / n;

    // threads[0].do_task([&]{  
    //     for(size_t i = 0; i < block_size; ++i)
    //     {
    //         fn(i);
    //     }
    // });
    // threads[1].do_task([&]{  
    //     for(size_t i = block_size; i < block_size * 2; ++i)
    //     {
    //         fn(i);
    //     }
    // });
    // threads[2].do_task([&]{  
    //     for(size_t i = block_size * 2; i < block_size * 3; ++i)
    //     {
    //         fn(i);
    //     }
    // });

    size_t offset = 0;
    for(size_t i = 0; i < n - 1; ++i)
    {
        threads[i].do_task([&, offset]{  
            for(size_t j = offset; j < offset + block_size; ++j)
            {
                fn(j);
            }
        });
        offset += block_size;
    }

    // std::thread j1{[&]{  
    //     for(size_t i = 0; i < block_size; ++i)
    //     {
    //         fn(i);
    //     }
    // }};

    // std::thread j2{[&]{  
    //     for(size_t i = block_size; i < block_size * 2; ++i)
    //     {
    //         fn(i);
    //     }
    // }};

    // std::thread j3{[&]{  
    //     for(size_t i = block_size * 2; i < block_size * 3; ++i)
    //     {
    //         fn(i);
    //     }
    // }};

    
    for(size_t i = block_size * (n - 1); i < size; ++i)
    {
        fn(i);
    }

    // threads[0].wait_idle();
    // threads[1].wait_idle();
    // threads[2].wait_idle();
    for(auto& thread : threads)
    {
        thread.wait_idle();
    }

    // j1.join();
    // j2.join();
    // j3.join();
}

void minist_test()
{
    using Node = Neuron<Sigmoid<>>;
    
    std::vector<float> v1{ 3 };//这玩意是只有一个元素的vector，那个元素的值是3
    std::vector<float> v2( 3 );//这玩意是3个元素的vector

    //Node::

    //init random
    std::cout << "input seed: ";
    unsigned int seed;
    std::cin >> seed;
    std::default_random_engine random_engine{ seed };
    std::uniform_real_distribution<float> uniform_dist{ -1.0f, 1.0f };

    //read training set
    auto train_set = read_minist_set("MNIST/train-labels.idx1-ubyte", "MNIST/train-images.idx3-ubyte");

    //init network.
    std::vector<Node> input_layer{ train_set.image_size };
    
    std::cout << "input hidden node count: ";
    size_t hidden_node_count; std::cin >> hidden_node_count;

    std::vector<Node> hidden_layer{ hidden_node_count };

    std::array<Node, 10> output_layer{};

    for(auto& hidden_node : hidden_layer)
    {
        for(auto& input_node : input_layer)
        {
            Node::connect(input_node, hidden_node, uniform_dist(random_engine));
        }
        for(auto& output_node : output_layer)
        {
            Node::connect(hidden_node, output_node, uniform_dist(random_engine));
        }
    }

    //train
    std::cout << "input learning rate: ";
    float learning_rate; std::cin >> learning_rate;

    size_t learning_cycle_count = 0;
    size_t correct_count;
    size_t last_correct_count = 0;
    do
    {
        ++learning_cycle_count;
        correct_count = 0;

        system("PAUSE");
        auto begin = std::chrono::system_clock::now();

        for(size_t i = 0; i < train_set.count; ++i)
        {
            size_t image_offset = train_set.image_size * i;
            
            for_each(input_layer.size(), [&](size_t j){
                input_layer[j].value() = train_set.images[image_offset + j] / 255.0f;
            });
            // for(size_t j = 0; j < input_layer.size(); ++j)
            // {
            //     input_layer[j].value() = train_set.images[image_offset + j] / 255.0f;
            // }
            
            for_each(hidden_layer.size(), [&](size_t j){
                hidden_layer[j].update();
            });
            // for(auto& hidden_node : hidden_layer)
            // {
            //     hidden_node.update();
            // }

            std::array<float, 10> targets{};
            targets[train_set.labels[i]] = 1.0f;
            
            unsigned int result = 0;
            float max_value = 0.0f;

            for(unsigned int k = 0; k < 10; ++k)
            {
                output_layer[k].update();
                if(output_layer[k].value() > max_value)
                {
                    max_value = output_layer[k].value();
                    result = k;
                }
                float error = output_layer[k].value() - targets[k];
                output_layer[k].value() = 2.0f * error * derivation(Sigmoid<>{})(output_layer[k].input());
            }


            if(result == train_set.labels[i])
            {
                ++correct_count;
            }

            for_each(hidden_layer.size(), [&](size_t j){
                hidden_layer[j].inverse_update(learning_rate);
            });
            // for(auto& hidden_node : hidden_layer)
            // {
            //     hidden_node.inverse_update(learning_rate);
            // }

            for_each(input_layer.size(), [&](size_t j){
                input_layer[j].inverse_update(learning_rate);
            });
            // for(auto& input_node : input_layer)
            // {
            //     input_node.inverse_update(learning_rate);
            // }
        }

        auto end = std::chrono::system_clock::now();
        system("PAUSE");
        std::cout << "spend " << std::chrono::duration<double>(end - begin).count() << "s\n";          

        if(correct_count > last_correct_count)
        {
            std::cout << "correct count: " << correct_count << ", learning_cycle_count: " << learning_cycle_count << '\n';
        }
    }while(correct_count < train_set.count && learning_cycle_count < 60);

    std::puts("learning complete.");

    //test
    auto test_set = read_minist_set("MNIST/t10k-labels.idx1-ubyte", "MNIST/t10k-images.idx3-ubyte");

    correct_count = 0;

    for(size_t i = 0; i < test_set.count; ++i)
    {
        size_t image_offset = test_set.image_size * i;
        for(size_t j = 0; j < input_layer.size(); ++j)
        {
            input_layer[j].value() = test_set.images[image_offset + j] / 255.0f;
        }
        for(auto& hidden_node : hidden_layer)
        {
            hidden_node.update();
        }

        std::array<float, 10> targets{};
        targets[test_set.labels[i]] = 1.0f;
        
        unsigned int result = 0;
        float max_value = 0.0f;

        for(unsigned int k = 0; k < 10; ++k)
        {
            output_layer[k].update();
            if(output_layer[k].value() > max_value)
            {
                max_value = output_layer[k].value();
                result = k;
            }
        }
        if(result == test_set.labels[i])
        {
            ++correct_count;
        }
    }
    std::cout << "correct count: " << correct_count << '\n';
}

void minist_test2()
{
    using Layer = neuronjh::FullyConnectedLayer<float, Sigmoid<>>;

    //init random
    std::cout << "input seed: ";
    unsigned int seed;
    std::cin >> seed;
    std::default_random_engine random_engine{ seed };
    std::uniform_real_distribution<float> uniform_dist{ -1.0f, 1.0f };
    auto random_generator = [&](){ return uniform_dist(random_engine); };

    //read training set
    auto train_set = read_minist_set("MNIST/train-labels.idx1-ubyte", "MNIST/train-images.idx3-ubyte");

    //init network.
    std::cout << "input hidden node count: ";
    size_t hidden_node_count; std::cin >> hidden_node_count;

    auto hidden_layer = Layer{ train_set.image_size, hidden_node_count, {}, random_generator };
    auto output_layer = Layer{ hidden_node_count, 10, {}, random_generator };

    std::vector<float> hidden_output(hidden_layer.output_size());
    std::vector<float> hidden_gardient(hidden_layer.output_size());
    std::vector<float> output_output(output_layer.output_size());
    std::vector<float> output_gardient(output_layer.output_size());

    //train
    std::cout << "input learning rate: ";
    float learning_rate; std::cin >> learning_rate;

    size_t learning_cycle_count = 0;
    size_t correct_count;
    size_t last_correct_count = 0;
    do
    {
        ++learning_cycle_count;
        correct_count = 0;

        system("PAUSE");
        auto begin = std::chrono::system_clock::now();

        for(size_t i = 0; i < train_set.count; ++i)
        {
            size_t image_offset = train_set.image_size * i;
            auto image_data = std::span<float>(train_set.images.begin() + image_offset, train_set.image_size);

            for_each(hidden_layer.output_size(), [&, outputs_with_gradients = hidden_layer.forward_with_gradient(image_data)](size_t j){
                auto[output, gradient] = outputs_with_gradients[j];
                hidden_output[j] = output;
                hidden_gardient[j] = gradient;
            });

            for_each(output_layer.output_size(), [&, outputs_with_gradients = output_layer.forward_with_gradient(hidden_output)](size_t j){
                auto[output, gradient] = outputs_with_gradients[j];
                output_output[j] = output;
                output_gardient[j] = gradient;
            });
            // for(auto& hidden_node : hidden_layer)
            // {
            //     hidden_node.update();
            // }

            std::array<float, 10> targets{};
            targets[train_set.labels[i]] = 1.0f;
            
            unsigned int result = 0;
            float max_value = 0.0f;

            for(unsigned int k = 0; k < 10; ++k)
            {
                if(output_output[k] > max_value)
                {
                    max_value = output_output[k];
                    result = k;
                }
                float error = output_output[k] - targets[k];
                output_gardient[k] = 2.0f * error * output_gardient[k];
            }


            if(result == train_set.labels[i])
            {
                ++correct_count;
            }

            for_each(hidden_layer.output_size(), [&](size_t j){
                float inverse_input_sum = 0;
                for(size_t k = 0; k < output_gardient.size(); ++k)
                {
                    inverse_input_sum += output_gardient[k] * output_layer.weight(j, k);
                    output_layer.weight(j, k) -= learning_rate * hidden_output[j] * output_gardient[k];
                }
                hidden_gardient[j] = inverse_input_sum * hidden_gardient[j];
            });

            for_each(hidden_layer.output_size(), [&](size_t j){
                for(size_t k = 0; k < hidden_layer.input_size(); ++k)
                {
                    hidden_layer.weight(k, j) -= learning_rate * image_data[k] * hidden_gardient[j];
                }
            });
            
        }

        auto end = std::chrono::system_clock::now();
        system("PAUSE");
        std::cout << "spend " << std::chrono::duration<double>(end - begin).count() << "s\n";          

        if(correct_count > last_correct_count)
        {
            std::cout << "correct count: " << correct_count << ", learning_cycle_count: " << learning_cycle_count << '\n';
        }
    }while(correct_count < train_set.count && learning_cycle_count < 60);

    std::puts("learning complete.");

    //test
    // auto test_set = read_minist_set("MNIST/t10k-labels.idx1-ubyte", "MNIST/t10k-images.idx3-ubyte");

    // correct_count = 0;

    // for(size_t i = 0; i < test_set.count; ++i)
    // {
    //     size_t image_offset = test_set.image_size * i;
    //     for(size_t j = 0; j < input_layer.size(); ++j)
    //     {
    //         input_layer[j].value() = test_set.images[image_offset + j] / 255.0f;
    //     }
    //     for(auto& hidden_node : hidden_layer)
    //     {
    //         hidden_node.update();
    //     }

    //     std::array<float, 10> targets{};
    //     targets[test_set.labels[i]] = 1.0f;
        
    //     unsigned int result = 0;
    //     float max_value = 0.0f;

    //     for(unsigned int k = 0; k < 10; ++k)
    //     {
    //         output_layer[k].update();
    //         if(output_layer[k].value() > max_value)
    //         {
    //             max_value = output_layer[k].value();
    //             result = k;
    //         }
    //     }
    //     if(result == test_set.labels[i])
    //     {
    //         ++correct_count;
    //     }
    // }
    // std::cout << "correct count: " << correct_count << '\n';
}

int main()
{
    minist_test2();
    //minist_test();
    //unary_fn_test([](auto x){ return 0.5f * x * x; }, 0.0f, 1.1f, 0.1f, 0.00001f, 0.5f);
}