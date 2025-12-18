NVCC        := nvcc
NVCC_FLAGS  := -O3 -Xcompiler -Wall --std=c++11
LIBS        := -lcublas

SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build

SRCS        := $(wildcard $(SRC_DIR)/*.cu)
OBJS        := $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/obj/%.o)
TARGET      := $(BUILD_DIR)/gemm_tool

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Liaison de l'exécutable : $@"
	$(NVCC) $(OBJS) -o $@ $(LIBS)

$(BUILD_DIR)/obj/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)/obj
	@echo "Compilation de $<..."
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR)/obj:
	mkdir -p $(BUILD_DIR)/obj

clean:
	@echo "Suppression du dossier $(BUILD_DIR)..."
	rm -rf $(BUILD_DIR)

.PHONY: all clean