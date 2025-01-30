#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "layer.hpp"
#include "aux.hpp"

class CnnNetwork :public sf::Transformable, public sf::Drawable
{
private:
    sf::Vector2f networkSize; //size, in pixels, of the layer on the screen
    int LayerNumber; //Number of layers in the network

    sf::Font NetworkFont;
    
    Button addLayerButton;
    Button removeLayerButton;

public:
    void addLayer(int n, float radius, sf::Vector2f position);
    void removeLayer();

    NeuralNetwork NN;

    bool trainingMode = false;

    std::vector<CNNLayer*> Layers;

    //Constructors
    CnnNetwork();
    CnnNetwork(int n, sf::Vector2f renderSize);

    //Destructor
    ~CnnNetwork();

    //setters
    void setFont(sf::Font& font){
        NetworkFont= font;
    }

    void setSize(const sf::Vector2f size);
    void setSize(const float x, const float y);

    //getters
    sf::Vector2f getSize() const;
    int getLayerNumber() ;
    int* getNeuronNumbers() const ;

    //Functions, methods
    bool handleClick(const sf::Vector2f mousePos);
    bool handleClick(const sf::Event& event); //overload to accept an event and make the main function prettier

    sf::Vector2f getNeuronPos(const CNNLayer& layer, const int i) const; //find the position of the i-th neuron of the layer
    void drawWeightPair(sf::RenderTarget& target, sf::RenderStates states, CNNLayer& layer1, CNNLayer& layer2, int neuron1, int neuron2,double weight = 0) const ;
    
    void drawWeights(sf::RenderTarget& target, sf::RenderStates states) const;

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};


#endif