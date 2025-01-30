#include "../include/layer.hpp"
#include "../include/aux.hpp"
#include "../include/network.hpp"
#include "../include/queue.hpp"
#include "../include/matplotlib.hpp"
#include <iostream>
#include <unistd.h>
#include <math.h>
#include <memory>
#include <thread>

#define WINDOWRATIO 0.8
#define NETWORKSCALING 0.6f
#define NETWORKPOSRATIO 0.2f


// FUNCTION NAME LINE 101

int main()
{

    //MULTI THREADING ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    ThreadSafeQueue< NeuralNetwork > networkQueue; //The backend provides the weights to display here
    

    //Create the window for rendering (no resize because it's hard to manage)
    const sf::VideoMode DesktopMode = sf::VideoMode::getDesktopMode(); // Get the desktop resolution
    sf::RenderWindow window(sf::VideoMode(DesktopMode.width*WINDOWRATIO,DesktopMode.height*WINDOWRATIO), "CNN Visualizer",sf::Style::Titlebar | sf::Style::Close);
    sf::Vector2f windowSize=sf::Vector2f(window.getSize());
    //Load external resources (fonts, textures)
    sf::Font mainfont;
    
    if(!mainfont.loadFromFile("Resources/Vogue.ttf")){
        std::cout<< "font error"<< std::endl;
    }   

    //Create the necessary shapes
    

        //A network of 3 layers
        CnnNetwork network(3, NETWORKSCALING*windowSize); 
        network.setPosition(NETWORKPOSRATIO*windowSize);
        //A box to put the network in
        sf::RectangleShape networkOutline(NETWORKSCALING*windowSize);
        networkOutline.setPosition(NETWORKPOSRATIO*windowSize);
        networkOutline.setFillColor(sf::Color::White);
        networkOutline.setOutlineColor(sf::Color::Black);
        networkOutline.setOutlineThickness(5);

        //The title
        sf::Text MainTitle("Neural Network: Visualizer", mainfont);
        MainTitle.setCharacterSize(60.f);
        MainTitle.setStyle(sf::Text::Bold);
        MainTitle.setFillColor(sf::Color::Black);

        MainTitle.setOrigin(sf::Vector2f(0.5f*getTextSize(MainTitle).x,0)); // Place the origin in the middle at the top of the textbox
        MainTitle.setPosition(sf::Vector2f(0.5f*windowSize.x,0.f));


        //The parameters
        ButtonList paramButtons(windowSize, mainfont);

        //The training button
        Button trainingButton( sf::Vector2f(300,40),"Start training", mainfont);
        trainingButton.setOrigin(0.5f* trainingButton.getSize());
        trainingButton.setPosition(sf::Vector2f(0.5f*windowSize.x, 0.9f*windowSize.y));

        //The image to display at the end
        sf::Texture textureOutput;
        sf::Sprite Output;

        //Plotter
        AxesPlotter plotter;

        //trainingLoss
        sf::Text lossText;

    while (window.isOpen())//Main loop////////////////////////////////////////////////////////////////////////////////////
    {
        bool isUpdated =true;
        bool trained = false;
        sf::Event event;

        while (window.pollEvent(event)) //Event loop //////////////////////////////////////////////////////////////////////////
        {

            if (event.type == sf::Event::Closed)
                    window.close();

            if (event.type == sf::Event::MouseButtonPressed ){
                isUpdated = network.handleClick(event);// Network modification buttons
                if(paramButtons.whoIsClicked(event) != -1 ){
                    isUpdated= true;
                }
                paramButtons.handleButtonClick(paramButtons.whoIsClicked(event));  

                if(trainingButton.isClicked(event)){
                    isUpdated = true;
                    trained = true;
                    //BIG NEWS: TRAINING STARTS
                    trainingButton.setText("Training in progress...");
                    window.draw(trainingButton);
                    window.display();
                    std::cout<< "LayerNumber:"<<network.getLayerNumber()<<std::endl;
                    int epoch_push = 50;
                    NeuralNetwork NN = training( //Call Gabriel's function with user-defined arguments   
                                        network.getLayerNumber(),
                                        network.getNeuronNumbers(),
                                        paramButtons.m_buttonInfos[0].options[paramButtons.m_buttonInfos[0].currentOptionIndex],
                                        paramButtons.m_buttonInfos[1].options[paramButtons.m_buttonInfos[1].currentOptionIndex],
                                        std::stod(paramButtons.m_buttonInfos[2].options[paramButtons.m_buttonInfos[2].currentOptionIndex]),
                                        std::stod(paramButtons.m_buttonInfos[3].options[paramButtons.m_buttonInfos[3].currentOptionIndex]),
                                        std::stoi(paramButtons.m_buttonInfos[4].options[paramButtons.m_buttonInfos[4].currentOptionIndex]),
                                        std::stoi(paramButtons.m_buttonInfos[5].options[paramButtons.m_buttonInfos[5].currentOptionIndex]),
                                        epoch_push,
                                        paramButtons.m_buttonInfos[6].options[paramButtons.m_buttonInfos[6].currentOptionIndex],
                                        paramButtons.m_buttonInfos[7].options[paramButtons.m_buttonInfos[7].currentOptionIndex],
                                        std::ref(networkQueue));   

                    double range =  NN.range();
                    //Placement of the plot
                    plotter= AxesPlotter(- range, range , - range, range);
                    plotter.setScale(sf::Vector2f(150/range,150/range));
                    plotter.setPosition(windowSize.x  -  0.5*plotter.getSize().x,0.5f*windowSize.y);

                    //Calculation of the texture of the zones
                    MatrixXd matrixToDraw = NN.prediction_matrix;
                    textureOutput = matrixToTexture(matrixToDraw);
                    Output.setTexture(textureOutput);

                    //Alignment of the plot and zones
                    sf::Vector2u texturesize= textureOutput.getSize();
                    sf::Vector2f plotSize= plotter.getSize();
                    Output.setScale(sf::Vector2f(plotSize.x/ float(texturesize.x),plotSize.y/ float(texturesize.y))); 
                    Output.setOrigin(sf::Vector2f(0.5*Output.getLocalBounds().width,0.5f*Output.getLocalBounds().height));
                    Output.setPosition(plotter.getPosition());

                    //Plot the points            
                    MatrixXd test_labels = NN.predict(NN.test_features);
                    plotter.setPoints(NN.test_features, NN.test_labels);

                    //Display the loss
                    lossText.setString(std::string("Accuracy: ") + std::to_string(NN.accuracy) );
                    lossText.setFont(mainfont); 
                    lossText.setStyle(sf::Text::Bold);
                    lossText.setFillColor(sf::Color::Black);
                    lossText.setCharacterSize(30);
                    lossText.setOrigin(lossText.getLocalBounds().width,0.5f * lossText.getLocalBounds().height);
                    lossText.setPosition(0.99*windowSize.x,0.25*windowSize.y);                        
                    
                    //Reset the button to normal
                    trainingButton.setText("Start Training");
                }
            }
        }    

        if(isUpdated){
            if (trained){      
                std::cout<<"TEXTURE UPDATED"<<std::endl;
                trained = false;
                isUpdated=false;
            }

            window.clear(sf::Color::White);

            //The little bunny cat
            window.draw(Output);
            window.draw(plotter);
            //The network
            window.draw(networkOutline);
            window.draw(network);

            //The title
            window.draw(MainTitle);
            //The buttons
            window.draw(paramButtons);
            window.draw(trainingButton);
            //The loss
            window.draw(lossText);

            window.display();
        }
    }
    return 0;
}
