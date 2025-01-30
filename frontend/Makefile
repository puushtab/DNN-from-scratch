# Compilateur et flags
CXX := g++
CXXFLAGS := -g -std=c++20 -Iinclude -I../backend/include -Wall -Wextra -MMD -MP
LDFLAGS :=
LDLIBS := -lsfml-graphics -lsfml-window -lsfml-system

# Répertoires
BUILD_DIR := compilation
SRC_DIR := src
BACKEND_SRC_DIR := ../backend/src

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
BACKEND_FILES := $(wildcard $(BACKEND_SRC_DIR)/*.cpp)

OBJS := \
	$(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES)) \
	$(patsubst $(BACKEND_SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(BACKEND_FILES))

DEPFILES := $(OBJS:.o=.d)

# Cible principale
TARGET := sfml-app

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# Règle générique pour la compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Règle générique pour la compilation des fichiers backend
$(BUILD_DIR)/%.o: $(BACKEND_SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
# Inclusion des dépendances générées
-include $(DEPFILES)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
