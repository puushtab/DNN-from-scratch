# DNN-from-scratch in C++

This project is a C++ application with a graphical user interface (GUI) that allows users to customize the hyperparameters and structure of a neural network to classify 2D points. The application provides real-time visualization of classification results, enabling users to experiment with various configurations interactively.

---
## **Screenshots**
![image](https://github.com/user-attachments/assets/f7ddbe53-5fa2-45be-b641-650cbe111e0d)
![image](https://github.com/user-attachments/assets/be666087-b3bb-404a-96a6-1efad3e22674)

---

## **Features**
- Customizable neural network architecture (e.g., number of layers, neurons per layer, activation functions).
- Adjustable hyperparameters (e.g., learning rate, batch size, epochs).
- Real-time 2D point classification visualization.
- Interactive GUI to modify settings without code changes.
- A backend that is fully flexible and easy-to-use
---

### **Prerequisites**
- C++ compiler supporting C++20 (or later)
- [SFML](https://www.sfml-dev.org/fr/) 
- [Eigen](https://eigen.tuxfamily.org/) (for matrix operations).

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/puushtab/DNN-from-scratch.git
   cd DNN-from-scratcg
   ```
2. Configure and build the project:
   ```bash
  make
   ```

3. Run the application:
   ```bash
  ./sfml-application
   ```
---

## **Good-to-know**
- The visual representation is representing test points among the "classification zone". Every point outside of the zone of its color is misclassified.
- If the GUI freezes, it is normal.
- This project will evolve and features will be added through time such as parallelism, non-blocking GUI and more loss hyperparameters to play with.
  
---


### **Hyperparameters**
- Loss function: Loss function, fixed for the moment (MSE)
- Activation: Activation function applied at each layer, ReLU or sigmoid. At the last layer of ReLU, a sigmoid is applied.
- Learning rate: Adjusts the step size during training. It is fixed for now.
- Training ratio: The ratio of training data to total data
- Batch size: Number of samples processed in each training iteration. The size determines the optimizer, 700 = BGD, 1 = SGD, other = MBGD
- Number of epochs: Number of complete passes through the dataset.
- Optimizer: Fixed, MBGD-like and chosen by the batch size
- Dataset: Dataset to classify (Moons, Blobs, Linear or Circles)

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push your branch and submit a pull request.

---

## **License**
_This project is under the MIT License_
---

## **Acknowledgments**
_(Mention any libraries, frameworks, or individuals who contributed to the project.)_
- Maloe Aymonier: Worked on the frontend of this project
- Gabriel Dupuis: Worked on the backend of this project
