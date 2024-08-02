#include <concepts>
#include <iostream>
#include <random>
#include <array>
#include <fstream>
#include <ranges>
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
        std::vector<unsigned char> images;
    };

    std::cout << "read " << labels.size() << " data.\n";

    return ResultType{ labels.size(), image_size, std::move(labels), std::move(images) };
}

void minist_test()
{
    using Node = Neuron<Sigmoid<>>;
    
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

        for(size_t i = 0; i < train_set.count; ++i)
        {
            size_t image_offset = train_set.image_size * i;
            for(size_t j = 0; j < input_layer.size(); ++j)
            {
                input_layer[j].value() = train_set.images[image_offset + j] / 255.0f;
            }
            for(auto& hidden_node : hidden_layer)
            {
                hidden_node.update();
            }

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

            for(auto& hidden_node : hidden_layer)
            {
                hidden_node.inverse_update(learning_rate);
            }
            for(auto& input_node : input_layer)
            {
                input_node.inverse_update(learning_rate);
            }
        }
        if(correct_count > last_correct_count)
        {
            std::cout << "correct count: " << correct_count << ", learning_cycle_count: " << learning_cycle_count << '\n';
        }
    }while(correct_count < train_set.count && learning_cycle_count <= 10);

    std::puts("learning complete.");

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

int main()
{
    minist_test();
    //unary_fn_test([](auto x){ return 0.5f * x * x; }, 0.0f, 1.1f, 0.1f, 0.00001f, 0.5f);
}