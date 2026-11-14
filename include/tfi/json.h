// Small JSON reader, enough for safetensors headers and indranet configs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tfi::json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::string> keys;  // object keys, in file order
    std::vector<Value> values;      // object values, parallel to keys

    bool is_null() const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }
    bool is_bool() const { return type == Type::Bool; }

    const Value* find(const std::string& key) const;  // nullptr if absent
    const Value& at(const std::string& key) const;    // throws if absent
    double as_number() const;
    int64_t as_int() const;
    bool as_bool() const;
    const std::string& as_string() const;
};

Value parse(const std::string& text);

}  // namespace tfi::json
