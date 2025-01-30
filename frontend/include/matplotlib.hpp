#pragma once

#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <vector>

class AxesPlotter : public sf::Drawable, public sf::Transformable
{
public:
    AxesPlotter() = default;
    AxesPlotter(float xMin, float xMax, float yMin, float yMax, 
                sf::Color axesColor = sf::Color::Black);

    void setPoints(const Eigen::MatrixXd& points, 
                   const Eigen::MatrixXd colors = {});

    sf::Vector2f getSize() const ;

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    float m_xMin, m_xMax, m_yMin, m_yMax;
    float m_baseRadius; // Base size of the circles
    sf::VertexArray m_axes;
    std::vector<sf::CircleShape> m_circles; // Container for the circles
};