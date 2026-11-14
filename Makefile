# Build the tfi engine (library), the indra CLI and the test tools with make (no CMake needed).
#   make            -> build/libtfi.a build/indra build/tfi_golden build/tfi_tests
#   make test       -> C++ unit tests (golden tests against PyTorch run from pytest)
# CXXFLAGS set on the command line adds to the required flags instead of replacing them:
#   make CXXFLAGS="-O1 -g -fsanitize=address,undefined" test
CXXFLAGS ?= -O2
override CXXFLAGS += -std=c++17 -Wall -Wextra -Iinclude -pthread
LDLIBS :=

BUILD := build
LIB_SRCS := $(wildcard src/*.cpp)
LIB_OBJS := $(LIB_SRCS:src/%.cpp=$(BUILD)/obj/%.o)

# macOS 15 or newer: Accelerate for matrix multiplies, Metal for the GPU path of batched patching (src/gpu.mm).
ifeq ($(shell uname -s),Darwin)
  override CXXFLAGS += -DTFI_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -mmacosx-version-min=15.0
  LDLIBS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
  LIB_OBJS += $(BUILD)/obj/gpu.o
endif
HEADERS := $(wildcard include/tfi/*.h src/*.h)

all: $(BUILD)/libtfi.a $(BUILD)/indra $(BUILD)/tfi_golden $(BUILD)/tfi_tests

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/%.o: src/%.mm $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -fobjc-arc -c $< -o $@

$(BUILD)/libtfi.a: $(LIB_OBJS)
	ar rcs $@ $^

$(BUILD)/indra: tools/indra.cpp $(BUILD)/libtfi.a $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libtfi.a $(LDLIBS) -o $@

$(BUILD)/tfi_golden: tests/golden.cpp $(BUILD)/libtfi.a $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libtfi.a $(LDLIBS) -o $@

$(BUILD)/tfi_tests: tests/unit.cpp $(BUILD)/libtfi.a $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libtfi.a $(LDLIBS) -o $@

test: $(BUILD)/tfi_tests
	$(BUILD)/tfi_tests

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
