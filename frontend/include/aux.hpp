#ifndef AUX_HPP
#define AUX_HPP
#include "../../backend/include/backend.hpp"
#include <SFML/Graphics.hpp>
#include <memory>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include <array>
#include <Eigen/Dense>

// Button Class ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Button : public sf::Drawable, public sf::Transformable {
    public:
        sf::RectangleShape m_rectangle;
        sf::Text m_text;
        
        void centerText() ;

        Button();
        Button(sf::Vector2f size, const std::string& text, const sf::Font& font);
        // Main draw override
        
        void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

        bool isClicked(const sf::Vector2f& mousePos) const;
        bool isClicked(const sf::Event event) const;

        // Setters with auto-centering
        void setSize(sf::Vector2f size);
        void setText(const std::string& text);
        void setFont(const sf::Font& font);

        // Getters
        sf::Vector2f getSize() const ;
        const sf::Text& getText() const ;
        const sf::RectangleShape& getShape() const;
};

//ButtonsList Class ////////////////////////////////////////////////////////////////////////////////:

class ButtonList : public sf::Drawable, public sf::Transformable {
private:
    struct ButtonInfo {
        std::string title;
        std::vector<std::string> options;
        int currentOptionIndex;
    };

    Button m_buttonLossFunction;
    Button m_buttonActivation;
    Button m_buttonLearningRate;
    Button m_buttonTrainingRatio;
    Button m_buttonBatchSize;
    Button m_buttonEpochs;
    Button m_buttonOptimizer;
    Button m_buttonDataset;

    void initializeButtons(const sf::Vector2f& windowSize, const sf::Font& font);
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
public:
    ButtonList(const sf::Vector2f& windowSize, const sf::Font& font);
    virtual ~ButtonList();

    const Button& getButtonLossFunction() const;
    const Button& getButtonActivation() const;
    const Button& getButtonLearningRate() const;
    const Button& getButtonTrainingRatio() const;
    const Button& getButtonBatchSize() const;
    const Button& getButtonEpochs() const;
    const Button& getButtonOptimizer() const;
    const Button& getButtonDataset() const;


    void setPosition(const sf::Vector2f& position);
    sf::Vector2f calculateAdjustedSize(const sf::Text& text) const;

    int whoIsClicked(const sf::Vector2f& mousePos) const; 
    int whoIsClicked(const sf::Event& event) const;
    void handleButtonClick(int buttonIndex); // Method to deal with parameter changes

    std::array<ButtonInfo, 8> m_buttonInfos; // Storage for title, index and text
};


template <typename T>
T dist(sf::Vector2<T> a, sf::Vector2<T> b);

sf::Vector2f getTextSize(const sf::Text& text) ;
void centerOrigin(sf::Text& text);
sf::Texture matrixToTexture(const Eigen::MatrixXd& matrix);

#endif

