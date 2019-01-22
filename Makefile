# directiories
BIN_DIR=bin
BUILD_DIR=build
INC_DIR=include
SRC_DIR=src
__IGNORE__ := $(shell mkdir -p $(BIN_DIR) $(BUILD_DIR) $(INC_DIR) $(SRC_DIR))

TARGET=main

# sources
CXX_SOURCES := $(shell find $(SRC_DIR) -name *.cpp)
NVC_SOURCES := $(shell find $(SRC_DIR) -name *.cu)

CXX_OBJS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
NVC_OBJS = $(NVC_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.cu.o)

# compilers
override CXX_FLAGS += -Wall -Wextra -std=c++11
CXX_LIBS = -I./$(INC_DIR)
CXX=g++

override NVC_FLAGS += --compiler-options -Wall,-Wextra -std=c++11
NVC_LIBS = -I./$(INC_DIR)
NVC=nvcc

override LD_FLAGS += --compiler-options -Wall,-Wextra -std=c++11
LD_LIBS = -I./$(INC_DIR) -lGL -lGLU -lglut -lGLEW -lgmp
LD=nvcc

all: $(BIN_DIR)/$(TARGET)

release:
	make all CXX_FLAGS="-O3" NVC_FLAGS="-O3" LD_FLAGS="-O3"

$(BIN_DIR)/$(TARGET): $(CXX_OBJS) $(NVC_OBJS)
	$(LD) $^ $(LD_FLAGS) -o $@ $(LD_LIBS)

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVC) $< $(NVC_FLAGS) -c -o $@ $(NVC_LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $< $(CXX_FLAGS) -c -o $@ $(CXX_LIBS)

clean:
	rm -fdr $(BUILD_DIR)
	rm -fdr $(BIN_DIR)
