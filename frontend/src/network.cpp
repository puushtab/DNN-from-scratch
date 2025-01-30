#include "../include/network.hpp"

float distf(sf::Vector2f a, sf::Vector2f b){
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

// Class CNNNetwork ///////////////////////////////////////////////
// Constructors
CnnNetwork::CnnNetwork(){
    LayerNumber = 0;
}
CnnNetwork::CnnNetwork(int n, sf::Vector2f renderSize):  networkSize(renderSize) ,LayerNumber(n)
{
    NetworkFont.loadFromFile("Resources/Vogue.ttf");
    // Creation of buttons
    sf::Vector2f ButtonSize=sf::Vector2f(30,30);
    addLayerButton = Button(ButtonSize,"+",NetworkFont);
    removeLayerButton = Button(ButtonSize,"-",NetworkFont);

    // sf::Vector2f ButtonGlobalSize=getTransform().transformPoint(ButtonSize);
    addLayerButton.setPosition(getSize().x -0.5f*ButtonSize.x , 0.5*networkSize.y-ButtonSize.y);
    removeLayerButton.setPosition(getSize().x  -0.5f*ButtonSize.x, 0.5*networkSize.y);
    
    // Creation of layers
    Layers.reserve(n);
    // Create the first layer with 2 inputs
    Layers.push_back(new CNNLayer(2,10,renderSize.y,sf::Vector2f(0,0)));
    Layers[0]->setFont(NetworkFont);
    for (int i = 1; i < n-1; ++i) {
        Layers.push_back(new CNNLayer(3,10,renderSize.y,sf::Vector2f(0,0))); // The default layer has 1 neuron and is at position 0. We will move them when drawing
        Layers[i]->setFont(NetworkFont);
    }
    // Create the last layer with one output
    Layers.push_back(new CNNLayer(1,10,renderSize.y,sf::Vector2f(0,0)));
    Layers[n-1]->setFont(NetworkFont);
}
// Destructor
CnnNetwork::~CnnNetwork(){
    for (int i = 0; i < LayerNumber; ++i) {
        delete Layers[i];
    }
}

// Setters
 void CnnNetwork::setSize(const sf::Vector2f size){
     networkSize=size;
}
void CnnNetwork::setSize(const float x, const float y){
    networkSize= sf::Vector2f(x,y);
}
// Getters
sf::Vector2f CnnNetwork::getSize() const {
    return networkSize;
}
int CnnNetwork::getLayerNumber() { return LayerNumber-1;}

int* CnnNetwork::getNeuronNumbers() const {
    int* numbers = new int[LayerNumber-1];
    for (int i = 1 ; i < LayerNumber ; ++i){
        numbers[i-1] = Layers[i]->getNeuronNumber();
        std::cout << "Numbers "<< i-1 << " : " << numbers[i-1]<<std::endl;
    }
    return numbers;
}

// Functions, methods

bool CnnNetwork::handleClick(const sf::Vector2f mousePos){

    // Transform mouse position to local coordinates
    sf::Vector2f localMouse = getInverseTransform().transformPoint(mousePos);

    for (int layer=0 ; layer < LayerNumber-1;++layer) {
        if (layer!=0 && layer!= LayerNumber-1 && Layers[layer]->handleClick(localMouse) ) {
            return true; // A layer was clicked, normally no button overlap
        }
    }

    if (addLayerButton.isClicked(localMouse)){
        addLayer(1, 10, sf::Vector2f(0,0));
        return true;
    }
    else{if(removeLayerButton.isClicked(localMouse)){
        removeLayer();
        return true;
    }
    }

    return false; // no click found
}
bool CnnNetwork::handleClick(const sf::Event &event){ // overload to directly handle an event
    return handleClick( sf::Vector2f(event.mouseButton.x,event.mouseButton.y));
}

void CnnNetwork::addLayer(int n, float radius, sf::Vector2f position){

    Layers.emplace(Layers.end()-1,new CNNLayer(n,radius,networkSize.y,position));
    Layers[LayerNumber-1]->setFont(NetworkFont);
    ++LayerNumber;
}

void CnnNetwork::removeLayer(){
    if(LayerNumber >2){
    Layers.erase(Layers.end()-2);
    --LayerNumber;
    }
}

sf::Vector2f CnnNetwork::getNeuronPos(const CNNLayer& layer, const int i) const { // find the position of the i-th neuron of the layer
   sf::Vector2f layerPos = layer.getNeuronPos(i);
    layerPos= getTransform().transformPoint(layerPos);
    layerPos-= getPosition();
    return layerPos;
}

void CnnNetwork::drawWeightPair(sf::RenderTarget& target, sf::RenderStates states, CNNLayer& layer1, CNNLayer& layer2, int neuron1, int neuron2,double weight) const {
    sf::Vector2f pos1 = getNeuronPos(layer1, neuron1);
    sf::Vector2f pos2 = getNeuronPos(layer2, neuron2);

   // Convert the weight (float between -1 and 1) to a thickness in px
   int width = abs(int(weight*5) );
   
    // Calculate the distance and check
    float dx = pos2.x - pos1.x;
    float dy = pos2.y - pos1.y;
    float length = std::hypot(dx, dy);
    if (length < 1e-5f) return;

    // Create the line (length, thickness)
    sf::RectangleShape line(sf::Vector2f(length, width));
    if ( weight >0 ) line.setFillColor(sf::Color::Blue);
    if ( weight <0 ) line.setFillColor(sf::Color::Red);

    // Positioning and rotation
    line.setOrigin(0, weight/2.0f); // Vertical centering
    line.setPosition(pos1);
    line.setRotation(atan2f(dy, dx) * (180.0f / M_PI));

    target.draw(line, states);
}

void CnnNetwork::drawWeights(sf::RenderTarget& target, sf::RenderStates states) const{ 
    int N = LayerNumber;
    if(trainingMode){
    
    for(int i_layer=0 ; i_layer < N-1; ++i_layer){
        assert(Layers[i_layer] != nullptr && "Layers[i_layer] is null");
        assert(Layers[i_layer + 1] != nullptr && "Layers[i_layer + 1] is null");   

        for(int neuron1=0; neuron1 < Layers[i_layer]->getNeuronNumber(); ++neuron1){
            for(int neuron2=0; neuron2 < Layers[i_layer+1]->getNeuronNumber(); ++neuron2){ // triple loop to select the interface between two layers, and a neuron in each layer
                std::cout << "Layer number: " << N << " Layer: " << i_layer << " neuron1: " << neuron1 << " neuron2: " << neuron2 << std::endl;
                
                NN.printDetails();
                
                // Assert that the i_layer index is within bounds
                assert(i_layer + 1 < NN.layers.size() && "Layer index out of bounds");
                std::cout << "???" << std::endl;

                const auto& layer = NN.layers[i_layer];
                std::cout << "Layers ok" << std::endl;

                // Assert that the neuron indices are within bounds
                assert(neuron2 < layer->weights.rows() && "Neuron2 index out of bounds");
                assert(neuron1 < layer->weights.cols() && "Neuron1 index out of bounds");
                std::cout << "???" << std::endl;
                std::cout << layer->weights << std::endl;

                double w = layer->weights(neuron2, neuron1); // transformation because we do not use the same notation system
                std::cout << "Access ok" << w << std::endl;
                drawWeightPair(target, states, *Layers[i_layer], *Layers[i_layer + 1], neuron1, neuron2, w);
            }
        }
    }
    }else{
        for(int i_layer=0 ; i_layer < N-1;++i_layer){
            for(int neuron1=0; neuron1< Layers[i_layer]->getNeuronNumber();++neuron1){
                for(int neuron2=0; neuron2 < Layers[i_layer+1]->getNeuronNumber();++neuron2){ // triple loop to select the interface between two layers, and a neuron in each layer
                    drawWeightPair(target,states,*Layers[i_layer],*Layers[i_layer+1],neuron1,neuron2,0.4);
                }
            }
        }
    }
}

void CnnNetwork::draw(sf::RenderTarget& target, sf::RenderStates states) const {

    sf::Vector2f networkSize = getSize(); 
    float offsetX = networkSize.x / (LayerNumber + 1);
    states.transform *= getTransform();

    drawWeights(target, states);

    target.draw(addLayerButton,states);
    target.draw(removeLayerButton,states);

    for(int i =0; i< LayerNumber; ++i){
        Layers[i]->setPosition(sf::Vector2f(offsetX*(i+1),0));
        target.draw(*Layers[i],states);
    }
}
