#include "../include/aux.hpp"

// Definition of the Button class functions
void Button::centerText() {
    sf::FloatRect textBounds = m_text.getLocalBounds();
    m_text.setOrigin(textBounds.left + textBounds.width/2.0f,
                    textBounds.top + textBounds.height/2.0f);
    m_text.setPosition(m_rectangle.getSize().x/2.0f, 
                    m_rectangle.getSize().y/2.0f);
}

Button::Button() = default;

Button::Button(sf::Vector2f size, const std::string& text, const sf::Font& font) 
    : m_rectangle(size), m_text(text, font) 
{
    // Rectangle styling
    m_rectangle.setFillColor(sf::Color::White);
    m_rectangle.setOutlineColor(sf::Color::Black);
    m_rectangle.setOutlineThickness(2.0f);
    
    // Text styling
    m_text.setFillColor(sf::Color::Black);
    m_text.setCharacterSize(static_cast<unsigned int>(size.y)); 
    m_text.setStyle(sf::Text::Style::Bold);
    centerText();
}

// Main draw override
void Button::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    states.transform *= getTransform();  // Apply button's transform
    target.draw(m_rectangle, states);
    target.draw(m_text, states);
}

bool Button::isClicked(const sf::Vector2f &mousePos) const {
    // The function returns true if the click is inside the rectangle.
    sf::Vector2f localMouse = getInverseTransform().transformPoint(mousePos);
    return m_rectangle.getLocalBounds().contains(localMouse);
}
bool Button::isClicked(const sf::Event event) const {
    return isClicked(sf::Vector2f(event.mouseButton.x,event.mouseButton.y));
}
// Setters with auto-centering
void Button::setSize(sf::Vector2f size) {
    m_rectangle.setSize(size);
    centerText();
}

void Button::setText(const std::string& text) {
    m_text.setString(text);
    centerText();
}

void Button::setFont(const sf::Font& font) {
    m_text.setFont(font);
}

// Getters
sf::Vector2f Button::getSize() const { return m_rectangle.getSize(); }
const sf::Text& Button::getText() const { return m_text; }
const sf::RectangleShape& Button::getShape() const { return m_rectangle; }

// ButtonList CLASS ////////////////////////////////////////////////////////////////////////////////

ButtonList::ButtonList(const sf::Vector2f& windowSize, const sf::Font& font) {
    // Initialization of parameters for each button
    m_buttonInfos = {{
        {"Loss function", {"MSE loss"}, 0},
        {"Activation", {"Sigmoid", "ReLU"}, 0},
        {"Learning rate", {"1","0.1", "0.01", "0.001"}, 0},
        {"Training ratio", {"0.7", "0.8", "0.9"}, 0},
        {"Batch size", {"1", "32", "64", "128", "700"}, 0},
        {"Epochs", {"100", "1000", "5000","10000", "50000"}, 0},
        {"Optimizer", {"MiniBatch"}, 0},
        {"Dataset", {"Moons", "Blobs", "Linear","Circles"}, 0}
    }};
    initializeButtons(windowSize, font);
}
ButtonList::~ButtonList()= default;

// Getters
const Button& ButtonList::getButtonLossFunction() const { return m_buttonLossFunction; }
const Button& ButtonList::getButtonActivation() const { return m_buttonActivation; }
const Button& ButtonList::getButtonLearningRate() const { return m_buttonLearningRate; }
const Button& ButtonList::getButtonTrainingRatio() const { return m_buttonTrainingRatio; }
const Button& ButtonList::getButtonBatchSize() const { return m_buttonBatchSize; }
const Button& ButtonList::getButtonEpochs() const { return m_buttonEpochs; }
const Button& ButtonList::getButtonOptimizer() const { return m_buttonOptimizer; }
const Button& ButtonList::getButtonDataset() const { return m_buttonDataset; }

void ButtonList::initializeButtons(const sf::Vector2f& windowSize, const sf::Font& font) {
    sf::Vector2f buttonSize(200.0f, 25.0f); // Height increased for 2 lines

    // Creation of buttons with title + first option
    m_buttonLossFunction = Button(buttonSize, m_buttonInfos[0].title + "\n" + m_buttonInfos[0].options[0], font);
    m_buttonActivation = Button(buttonSize, m_buttonInfos[1].title + "\n" + m_buttonInfos[1].options[0], font);
    m_buttonLearningRate = Button(buttonSize, m_buttonInfos[2].title + "\n" + m_buttonInfos[2].options[0], font);
    m_buttonTrainingRatio = Button(buttonSize, m_buttonInfos[3].title + "\n" + m_buttonInfos[3].options[0], font);
    m_buttonBatchSize = Button(buttonSize, m_buttonInfos[4].title + "\n" + m_buttonInfos[4].options[0], font);
    m_buttonEpochs = Button(buttonSize, m_buttonInfos[5].title + "\n" + m_buttonInfos[5].options[0], font);
    m_buttonOptimizer = Button(buttonSize, m_buttonInfos[6].title + "\n" + m_buttonInfos[6].options[0], font);
    m_buttonDataset = Button(buttonSize, m_buttonInfos[7].title + "\n" + m_buttonInfos[7].options[0], font);

    // Adjustment of button sizes
    auto adjustButton = [this](Button& btn, const ButtonInfo& info) {
        btn.setSize(calculateAdjustedSize(btn.getText()));
    };
    adjustButton(m_buttonLossFunction, m_buttonInfos[0]);
    adjustButton(m_buttonActivation, m_buttonInfos[1]);
    adjustButton(m_buttonLearningRate, m_buttonInfos[2]);
    adjustButton(m_buttonTrainingRatio, m_buttonInfos[3]);
    adjustButton(m_buttonBatchSize, m_buttonInfos[4]);
    adjustButton(m_buttonEpochs, m_buttonInfos[5]);
    adjustButton(m_buttonOptimizer, m_buttonInfos[6]);
    adjustButton(m_buttonDataset, m_buttonInfos[7]);

    setPosition(sf::Vector2f(10.0f, windowSize.y * 0.1f));
}

void ButtonList::handleButtonClick(int buttonIndex) {
    if (buttonIndex < 0 || buttonIndex >= (int)m_buttonInfos.size()) return;

    auto& info = m_buttonInfos[buttonIndex];
    info.currentOptionIndex = (info.currentOptionIndex + 1) % info.options.size();
    
    // Update the button text
    std::string newText = info.title + "\n" + info.options[info.currentOptionIndex];
    switch (buttonIndex) {
        case 0: m_buttonLossFunction.setText(newText); break;
        case 1: m_buttonActivation.setText(newText); break;
        case 2: m_buttonLearningRate.setText(newText); break;
        case 3: m_buttonTrainingRatio.setText(newText); break;
        case 4: m_buttonBatchSize.setText(newText); break;
        case 5: m_buttonEpochs.setText(newText); break;
        case 6: m_buttonOptimizer.setText(newText); break;
        case 7: m_buttonDataset.setText(newText); break;
    }

    // Center the text
    auto& button = [this, buttonIndex]() -> Button& {
        switch (buttonIndex) {
            case 0: return m_buttonLossFunction;
            case 1: return m_buttonActivation;
            case 2: return m_buttonLearningRate;
            case 3: return m_buttonTrainingRatio;
            case 4: return m_buttonBatchSize;
            case 5: return m_buttonEpochs;
            case 6: return m_buttonOptimizer;
            case 7: return m_buttonDataset;
            default: throw std::out_of_range("Invalid button index");
        }
    }();
    button.centerText();
}
int ButtonList::whoIsClicked(const sf::Vector2f& mousePos) const {
    if (m_buttonLossFunction.isClicked(mousePos)) return 0;
    if (m_buttonActivation.isClicked(mousePos)) return 1;
    if (m_buttonLearningRate.isClicked(mousePos)) return 2;
    if (m_buttonTrainingRatio.isClicked(mousePos)) return 3;
    if (m_buttonBatchSize.isClicked(mousePos)) return 4;
    if (m_buttonEpochs.isClicked(mousePos)) return 5;
    if (m_buttonOptimizer.isClicked(mousePos)) return 6;
    if (m_buttonDataset.isClicked(mousePos)) return 7;
    return -1;
}

int ButtonList::whoIsClicked(const sf::Event& event) const {
    if (event.type == sf::Event::MouseButtonPressed) {
        sf::Vector2f mousePos(
            static_cast<float>(event.mouseButton.x),
            static_cast<float>(event.mouseButton.y)
        );
        return whoIsClicked(mousePos);
    }
    return -1;
}

void ButtonList::setPosition(const sf::Vector2f& position) {
    const float spacing = 100.0f; // Adjust spacing if necessary
    m_buttonLossFunction.setPosition(position);
    m_buttonActivation.setPosition(position.x, position.y + spacing * 1);
    m_buttonLearningRate.setPosition(position.x, position.y + spacing * 2);
    m_buttonTrainingRatio.setPosition(position.x, position.y + spacing * 3);
    m_buttonBatchSize.setPosition(position.x, position.y + spacing * 4);
    m_buttonEpochs.setPosition(position.x, position.y + spacing * 5);
    m_buttonOptimizer.setPosition(position.x, position.y + spacing * 6);
    m_buttonDataset.setPosition(position.x, position.y + spacing * 7);
}

sf::Vector2f ButtonList::calculateAdjustedSize(const sf::Text& text) const {
    sf::FloatRect bounds = text.getLocalBounds();
    return {bounds.width + 40.0f, bounds.height + 0.0f};
}

void ButtonList::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    // Apply the transformation of the ButtonList
    states.transform *= getTransform();

    // Draw each button
    target.draw(m_buttonLossFunction, states);
    target.draw(m_buttonActivation, states);
    target.draw(m_buttonLearningRate, states);
    target.draw(m_buttonTrainingRatio, states);
    target.draw(m_buttonBatchSize, states);
    target.draw(m_buttonEpochs, states);
    target.draw(m_buttonOptimizer, states);
    target.draw(m_buttonDataset, states);
}

// USEFUL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T dist(sf::Vector2<T> a, sf::Vector2<T> b){
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

sf::Vector2f getTextSize(const sf::Text& text){
    sf::FloatRect bounds=text.getLocalBounds();
    return sf::Vector2f(bounds.width,bounds.height);
}

void centerOrigin(sf::Text& text){
    sf::Vector2f textsize = getTextSize(text);
    text.setOrigin(0.5f *textsize);
}

sf::Texture matrixToTexture(const Eigen::MatrixXd& matrix) {
     if(matrix.rows() != 1) {
        throw std::invalid_argument("The matrix must be of size 1xN²");
    }
    
    const int sizeSquared = matrix.cols();
    const int N = static_cast<int>(std::sqrt(sizeSquared));
    
    // Check that N² matches the size of the matrix
    if(N * N != sizeSquared) {
        throw std::invalid_argument("The size of the matrix is not a perfect square");
    }
    
    const int bufferSize = N * N * 4; // 4 components per pixel (RGBA)
    sf::Uint8* pixels = new sf::Uint8[bufferSize];

    // Fill by traversing an NxN grid
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            // Calculate the index in the 1xN² matrix (row-major order)
            const int matrixIndex = i * N + j;
            const double value = matrix(0, matrixIndex); // Access to row 0
            
            const int pixelIndex = (i * N + j) * 4;
            
            if(value < 0.5) {
                pixels[pixelIndex]     = 50;   // R
                pixels[pixelIndex + 1] = 100;   // G
                pixels[pixelIndex + 2] = 255; // B
                pixels[pixelIndex + 3] = 255; // A
            } else {
                pixels[pixelIndex]     = 255; // R
                pixels[pixelIndex + 1] = 100;   // G
                pixels[pixelIndex + 2] = 50;   // B
                pixels[pixelIndex + 3] = 255; // A
            }
        }
    }

    sf::Texture texture;
    if (!texture.create(N, N)) {
        delete[] pixels;
        throw std::runtime_error("Failed to create texture");
    }
    
    texture.update(pixels);
    delete[] pixels;

    return texture;
}