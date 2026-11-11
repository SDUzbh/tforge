# Build the tfi library, CLI and test tools with make (no CMake needed).
#   make            -> build/libtfi.a build/tfi build/tfi_golden build/tfi_tests
#   make test       -> C++ unit tests (golden tests against PyTorch run from pytest)
# CXXFLAGS set on the command line adds to the required flags instead of replacing them:
#   make CXXFLAGS="-O1 -g -fsanitize=address,undefined" test
CXXFLAGS ?= -O2
override CXXFLAGS += -std=c++17 -Wall -Wextra -Iinclude -pthread
LDLIBS :=

ifeq ($(shell uname -s),Darwin)
  override CXXFLAGS += -DTFI_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
  LDLIBS += -framework Accelerate
endif

BUILD := build
LIB_SRCS := $(wildcard src/*.cpp)
LIB_OBJS := $(LIB_SRCS:src/%.cpp=$(BUILD)/obj/%.o)
HEADERS := $(wildcard include/tfi/*.h src/*.h)

all: $(BUILD)/libtfi.a $(BUILD)/tfi $(BUILD)/tfi_golden $(BUILD)/tfi_tests

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/libtfi.a: $(LIB_OBJS)
	ar rcs $@ $^

$(BUILD)/tfi: tools/tfi.cpp $(BUILD)/libtfi.a $(HEADERS)
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
