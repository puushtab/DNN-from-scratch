#include "../include/matplotlib.hpp"

AxesPlotter::AxesPlotter(float xMin, float xMax, float yMin, float yMax, sf::Color axesColor)
    : m_xMin(xMin), m_xMax(xMax), m_yMin(yMin), m_yMax(yMax)
{
    // Calculate the size of the circles (1/20 of the smallest dimension)
    const float graphWidth = xMax - xMin;
    const float graphHeight = yMax - yMin;
    m_baseRadius = std::min(graphWidth, graphHeight) / 100.0f;

    // Configure the axes
    m_axes.setPrimitiveType(sf::Lines);
    
    // X Axis
    m_axes.append(sf::Vertex(sf::Vector2f(xMin, 0.0f), axesColor));
    m_axes.append(sf::Vertex(sf::Vector2f(xMax, 0.0f), axesColor));
    
    // Y Axis
    m_axes.append(sf::Vertex(sf::Vector2f(0.0f, yMin), axesColor));
    m_axes.append(sf::Vertex(sf::Vector2f(0.0f, yMax), axesColor));

    // Invert Y for Cartesian coordinates
    setScale(1.f, -1.f);
}

void AxesPlotter::setPoints(const Eigen::MatrixXd& points, const Eigen::MatrixXd colors)
{
    m_circles.clear();
    const int numPoints = points.cols();

    for(int i = 0; i < numPoints; ++i)
    {
        const float x = points(0, i);
        const float y = points(1, i);
        
        // Create the circle
        sf::CircleShape circle(m_baseRadius);
        circle.setOrigin(m_baseRadius, m_baseRadius); // Center the circle
        circle.setPosition(x, y);
        
        // Default style
        circle.setOutlineColor(sf::Color::Black);
        circle.setOutlineThickness(0.01f);
        
        // Custom color or white by default
        if(i < colors.size()) {
            if(colors(i) > 0.5){
            circle.setFillColor(sf::Color::Red);
            }else{
            circle.setFillColor(sf::Color::Blue);
            }
        } else {
            circle.setFillColor(sf::Color::White);
        }
        
        m_circles.push_back(circle);
    }
}

void AxesPlotter::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    states.transform *= getTransform();
    
    // Draw the axes
    target.draw(m_axes, states);
    
    // Draw all the circles
    for(const auto& circle : m_circles) {
        target.draw(circle, states);
    }
}

sf::Vector2f AxesPlotter::getSize() const {
        return  sf::Vector2f(getScale().x * (m_xMax-m_xMin)  ,  getScale().y * (m_yMax-m_yMin) );       
    }