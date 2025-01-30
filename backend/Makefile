# Makefile for Neural Network project

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Iinclude -std=c++20 -Wall -Wextra 

# Eigen library path (update this path if necessary)
EIGEN_PATH = include/

# Include Eigen library
INCLUDES = -I$(EIGEN_PATH)

# Source files
SRCS = main.cpp src/nn.cpp src/activations.cpp src/layers.cpp src/losses.cpp src/utils.cpp src/backend.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = neural_network

# Default target
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up object files and executable
clean:
	rm -f $(OBJS) $(EXEC)

# Phony targets
.PHONY: all clean