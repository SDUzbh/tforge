#include "tfi/json.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace tfi::json {

const Value* Value::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == key) return &values[i];
    return nullptr;
}

const Value& Value::at(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw std::runtime_error("JSON: missing key '" + key + "'");
    return *v;
}

double Value::as_number() const {
    if (type != Type::Number) throw std::runtime_error("JSON: expected a number");
    return number;
}

int64_t Value::as_int() const {
    const double d = as_number();
    if (d != std::floor(d)) throw std::runtime_error("JSON: expected an integer");
    return static_cast<int64_t>(d);
}

bool Value::as_bool() const {
    if (type != Type::Bool) throw std::runtime_error("JSON: expected a boolean");
    return boolean;
}

const std::string& Value::as_string() const {
    if (type != Type::String) throw std::runtime_error("JSON: expected a string");
    return string;
}

namespace {

class Parser {
  public:
    explicit Parser(const std::string& s) : s_(s) {}

    Value parse_document() {
        Value v = parse_value();
        skip_ws();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

  private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(i_) + ": " + msg);
    }

    void skip_ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }

    char peek() {
        skip_ws();
        if (i_ >= s_.size()) fail("unexpected end of input");
        return s_[i_];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++i_;
    }

    bool consume_literal(const char* lit) {
        size_t n = 0;
        while (lit[n]) ++n;
        if (s_.compare(i_, n, lit) == 0) {
            i_ += n;
            return true;
        }
        return false;
    }

    Value parse_value() {
        const char c = peek();
        Value v;
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            v.type = Value::Type::String;
            v.string = parse_string();
            return v;
        }
        if (consume_literal("true")) { v.type = Value::Type::Bool; v.boolean = true; return v; }
        if (consume_literal("false")) { v.type = Value::Type::Bool; return v; }
        if (consume_literal("null")) return v;
        if (consume_literal("NaN")) { v.type = Value::Type::Number; v.number = NAN; return v; }
        if (consume_literal("Infinity")) { v.type = Value::Type::Number; v.number = INFINITY; return v; }
        if (consume_literal("-Infinity")) { v.type = Value::Type::Number; v.number = -INFINITY; return v; }
        return parse_number();
    }

    Value parse_number() {
        const char* start = s_.c_str() + i_;
        char* end = nullptr;
        const double d = std::strtod(start, &end);
        if (end == start) fail("invalid value");
        i_ += static_cast<size_t>(end - start);
        Value v;
        v.type = Value::Type::Number;
        v.number = d;
        return v;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t parse_hex4() {
        if (i_ + 4 > s_.size()) fail("bad \\u escape");
        uint32_t cp = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = s_[i_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= h - '0';
            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
            else fail("bad \\u escape");
        }
        return cp;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            const char c = s_[i_++];
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (i_ >= s_.size()) fail("unterminated escape");
            const char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && s_.compare(i_, 2, "\\u") == 0) {
                        i_ += 2;
                        const uint32_t lo = parse_hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }

    Value parse_array() {
        expect('[');
        Value v;
        v.type = Value::Type::Array;
        if (peek() == ']') { ++i_; return v; }
        while (true) {
            v.array.push_back(parse_value());
            const char c = peek();
            ++i_;
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']'");
        }
        return v;
    }

    Value parse_object() {
        expect('{');
        Value v;
        v.type = Value::Type::Object;
        if (peek() == '}') { ++i_; return v; }
        while (true) {
            if (peek() != '"') fail("expected object key");
            v.keys.push_back(parse_string());
            expect(':');
            v.values.push_back(parse_value());
            const char c = peek();
            ++i_;
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}'");
        }
        return v;
    }
};

}  // namespace

Value parse(const std::string& text) { return Parser(text).parse_document(); }

}  // namespace tfi::json
