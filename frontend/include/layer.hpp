#ifndef LAYER_HPP
#define LAYER_HPP
#include <SFML/Graphics.hpp>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include <memory>
#include "aux.hpp"

class CNNLayer: public sf::Drawable, public sf::Transformable {
    private:
        class CNNNeuron: public sf::CircleShape{
            private:
                double value;
            public:
                CNNNeuron();
                CNNNeuron(double x);

                void setValue(double x);
                const double getValue();

        };

        int NeuronNumber;
        std::vector<CNNNeuron*> Neurons;

        float neuronRadius;
        sf::Font NeuronFont;
        float LayerHeight; // vertical size of the layer, to be multiplied by the window height to get the absolute size

    public:
        Button addButton;
        Button rmButton;

        //constructors, destructors
        CNNLayer();

        CNNLayer(int n);

        CNNLayer(int n, float radius, float height, sf::Vector2f position);

        ~CNNLayer();
        
        void setFont(sf::Font& font);
        sf::Font* getFont();

        void setNeuronRadius(float radius);
        float getNeuronRadius();

        int getNeuronNumber();

        //actually it's a bit of a lie because we modify the neuron but since we don't modify the pointer to this neuron it's fine (and it's very convenient)
        void setNeuronPos(const int i, const sf::Vector2f position) const;
        sf::Vector2f getNeuronPos(const int i) const;

        void addNeuron(double Value);
        void removeNeuron();

        bool handleClick(const sf::Vector2f& mousePos);
        
        void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};

#endif