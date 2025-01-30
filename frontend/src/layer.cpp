#include "../include/layer.hpp"

// Definition of the functions of the class CNNLayer::CNNeuron////////////////////////                                                             
CNNLayer::CNNNeuron::CNNNeuron() : value(0) {
    setOutlineColor(sf::Color::Black);
    setOutlineThickness(2);
}                                   
CNNLayer::CNNNeuron::CNNNeuron(double x): value(x) {
    setOutlineColor(sf::Color::Black);
    setOutlineThickness(2);
}                            

void CNNLayer::CNNNeuron::setValue(double x){
    value=x;
}
const double CNNLayer::CNNNeuron::getValue(){
    return value;
}
////////////////////////////////////////////////////////////////////////////////////
        

// Definition of the functions of the class CNNLayer////////////////////////////////
// Constructors, destructors
CNNLayer::CNNLayer() : NeuronNumber(0), Neurons(), neuronRadius(0), LayerHeight(0), addButton(Button()), rmButton(Button()){}

CNNLayer::CNNLayer(int n) : NeuronNumber(n), neuronRadius(0), LayerHeight(0) {
    Neurons.reserve(n); // Preallocate the correct size
    for (int i = 0; i < n; ++i) {
        Neurons.push_back(new CNNNeuron());
        Neurons[i]->setRadius(neuronRadius);
    }
}

CNNLayer::CNNLayer(int n, float radius, float height, sf::Vector2f position) : 
    NeuronNumber(n), neuronRadius(radius), LayerHeight(height),
    addButton(Button(sf::Vector2f(2 * neuronRadius, 2 * neuronRadius), "+", NeuronFont)),
    rmButton(Button(sf::Vector2f(2 * neuronRadius, 2 * neuronRadius), "-", NeuronFont)) 
{
    rmButton.move(0, 2 * neuronRadius);
    this->setPosition(position);            
    Neurons.reserve(n); // Preallocate the correct size
    for (int i = 0; i < n; ++i) {
        Neurons.push_back(new CNNNeuron());
        Neurons[i]->setRadius(neuronRadius);
        Neurons[i]->setOrigin(sf::Vector2f(neuronRadius, neuronRadius));
    }
}

CNNLayer::~CNNLayer() {
    for (int i = 0; i < NeuronNumber; ++i) {
        delete Neurons[i];
    }
}

void CNNLayer::setFont(sf::Font& font) {
    NeuronFont = font;
}
sf::Font* CNNLayer::getFont() {
    return &NeuronFont;
}
void CNNLayer::setNeuronRadius(float radius) {
    neuronRadius = radius;
    for (int i = 0; i < NeuronNumber; ++i) {
        Neurons[i]->setRadius(neuronRadius);
    }
}
float CNNLayer::getNeuronRadius() {
    return neuronRadius;
}

int CNNLayer::getNeuronNumber() {
    return NeuronNumber;
}

void CNNLayer::setNeuronPos(const int i, const sf::Vector2f position) const {
    Neurons[i]->setPosition(position);
}
sf::Vector2f CNNLayer::getNeuronPos(const int i) const {
    if (i < 0 || i >= (int)Neurons.size()) 
        return sf::Vector2f(-1, -1);
    sf::Vector2f localPos = Neurons[i]->getPosition();
    return getTransform().transformPoint(localPos); // Apply the inverse transform of the current object (the layer) so that the returned position is valid in the space of the member that will retrieve it
}

void CNNLayer::addNeuron(double Value) {
    ++NeuronNumber;
    Neurons.push_back(new CNNNeuron(Value));
    Neurons[Neurons.size() - 1]->setRadius(neuronRadius);
    Neurons[Neurons.size() - 1]->setOrigin(0.5f * sf::Vector2f(neuronRadius, neuronRadius));
}
void CNNLayer::removeNeuron() {
    if (NeuronNumber > 0) {
        --NeuronNumber;
        Neurons.pop_back();
    }
}

bool CNNLayer::handleClick(const sf::Vector2f& mousePos) {
    // Transform mouse position to local coordinates
    sf::Vector2f localMouse = getInverseTransform().transformPoint(mousePos);

    // Check if the add or remove button was clicked
    if (addButton.isClicked(localMouse)) {
        addNeuron(0); 
        return true;
    }
    if (rmButton.isClicked(localMouse)) {
        removeNeuron(); 
        return true;
    }
    return false;
}

void CNNLayer::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    states.transform *= getTransform();
    
    // Calculate the height and space between neurons
    float offsetY = LayerHeight / (NeuronNumber + 1);

    target.draw(addButton, states);
    target.draw(rmButton, states);

    // Create the text to display on the neurons
    sf::Text NeuronOutput;
    NeuronOutput.setFont(NeuronFont);
    NeuronOutput.setCharacterSize((int)3 * neuronRadius);  
    NeuronOutput.setFillColor(sf::Color::White);
    NeuronOutput.setOutlineColor(sf::Color::Black);
    NeuronOutput.setOutlineThickness(2);

    for (int i = 0; i < NeuronNumber; ++i) {
        NeuronOutput.setString(std::to_string(i));
        // Offset the neurons to the correct position (since we know the size of the rendering window)
        setNeuronPos(i, sf::Vector2f(0, offsetY * (i + 1)));
        NeuronOutput.setPosition(sf::Vector2f(0, offsetY * (i + 1)));

        target.draw(*Neurons[i], states);
    }
}
////////////////////////////////////////////////////////////////////////////////////
