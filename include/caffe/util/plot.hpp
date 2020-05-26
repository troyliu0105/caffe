/**
 * The MIT License
 *
 * Copyright (c) 2017 Fabio Massaioli
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <forward_list>
#include <iomanip>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// include/utils.hpp

namespace plot {

namespace utils {
template <typename T>
inline constexpr T min(T a, T b) {
  return (a <= b) ? a : b;
}

template <typename T>
inline constexpr T max(T a, T b) {
  return (a >= b) ? a : b;
}

template <typename T>
inline constexpr std::pair<T, T> minmax(T a, T b) {
  return (a <= b) ? std::pair<T, T>{a, b} : std::pair<T, T>{b, a};
}

template <typename T>
inline constexpr T clamp(T x, T mn, T mx) {
  return min(max(x, mn), mx);
}

template <typename T>
inline constexpr T sgn(T x) {
  return (x > T()) - (x < T());
}

template <typename T>
inline constexpr T abs(T x) {
  return (T() > x) ? -x : x;
}

template <typename T>
inline constexpr T gcd(T p, T q) {
  return q ? gcd(q, p % q) : abs(p);
}
} /* namespace utils */

} /* namespace plot */

// include/string_view.hpp

namespace plot {

using std::string_view;

namespace detail {
template <typename T>
inline bool contains(std::string_view haystack, T &&needle) {
  return haystack.find(std::forward<T>(needle)) != std::string_view::npos;
}
} /* namespace detail */

} /* namespace plot */

// include/point.hpp

namespace plot {

template <typename T>
struct GenericPoint;

template <typename T>
constexpr GenericPoint<T> operator+(GenericPoint<T> const &lhs,
                                    GenericPoint<T> const &rhs);

template <typename T>
constexpr GenericPoint<T> operator-(GenericPoint<T> const &lhs,
                                    GenericPoint<T> const &rhs);

template <typename T>
constexpr GenericPoint<T> operator*(GenericPoint<T> const &lhs, T const &rhs);

template <typename T>
constexpr GenericPoint<T> operator*(T const &lhs, GenericPoint<T> const &rhs);

template <typename T>
constexpr GenericPoint<T> operator/(GenericPoint<T> const &lhs, T const &rhs);

template <typename T>
constexpr GenericPoint<T> operator/(T const &lhs, GenericPoint<T> const &rhs);

template <typename T>
struct GenericPoint {
  using coord_type = T;

  constexpr GenericPoint() = default;

  constexpr GenericPoint(T coord_x, T coord_y) : x(coord_x), y(coord_y) {}

  constexpr T distance(GenericPoint const &other) const {
    return (other - *this).abs();
  }

  constexpr T abs() const { return std::sqrt(x * x + y * y); }

  constexpr GenericPoint clamp(GenericPoint const &min,
                               GenericPoint const &max) const {
    return {utils::clamp(x, min.x, max.x), utils::clamp(y, min.y, max.y)};
  }

  template <typename U>
  constexpr operator GenericPoint<U>() const {
    return {static_cast<U>(x), static_cast<U>(y)};
  }

  GenericPoint &operator+=(GenericPoint const &other) {
    return (*this) = (*this) + other;
  }

  GenericPoint &operator-=(GenericPoint const &other) {
    return (*this) = (*this) - other;
  }

  GenericPoint &operator*=(T n) { return (*this) = (*this) * n; }

  GenericPoint &operator/=(T n) { return (*this) = (*this) / n; }

  constexpr bool operator==(GenericPoint const &other) const {
    return x == other.x && y == other.y;
  }

  constexpr bool operator!=(GenericPoint const &other) const {
    return x != other.x || y != other.y;
  }

  T x = 0, y = 0;
};

template <typename T>
inline constexpr GenericPoint<T> operator+(GenericPoint<T> const &lhs,
                                           GenericPoint<T> const &rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

template <typename T>
inline constexpr GenericPoint<T> operator-(GenericPoint<T> const &lhs,
                                           GenericPoint<T> const &rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}

template <typename T>
inline constexpr GenericPoint<T> operator*(GenericPoint<T> const &lhs,
                                           T const &rhs) {
  return {lhs.x * rhs, lhs.y * rhs};
}

template <typename T>
inline constexpr GenericPoint<T> operator*(T const &lhs,
                                           GenericPoint<T> const &rhs) {
  return {lhs * rhs.x, lhs * rhs.y};
}

template <typename T>
inline constexpr GenericPoint<T> operator/(GenericPoint<T> const &lhs,
                                           T const &rhs) {
  return {lhs.x / rhs, lhs.y / rhs};
}

template <typename T>
inline constexpr GenericPoint<T> operator/(T const &lhs,
                                           GenericPoint<T> const &rhs) {
  return {lhs / rhs.x, lhs / rhs.y};
}

using Coord = std::ptrdiff_t;
using Coordf = float;

using Point = GenericPoint<Coord>;
using Pointf = GenericPoint<Coordf>;

template <typename T>
using GenericSize = GenericPoint<T>;

using Size = GenericSize<Coord>;
using Sizef = GenericSize<Coordf>;

} /* namespace plot */

// include/color.hpp

namespace plot {

struct Color;

struct Color32 {
  std::uint8_t r, g, b, a;
};

constexpr Color operator+(Color const &lhs, Color const &rhs);
constexpr Color operator-(Color const &lhs, Color const &rhs);
constexpr Color operator*(Color const &lhs, float const &rhs);
constexpr Color operator*(float const &lhs, Color const &rhs);
constexpr Color operator/(Color const &lhs, float const &rhs);
constexpr Color operator/(float const &lhs, Color const &rhs);

struct Color {
  constexpr Color() : r(0), g(0), b(0), a(0) {}

  constexpr Color(float cmp_r, float cmp_g, float cmp_b, float cmp_a = 1.0f)
      : r(cmp_r), g(cmp_g), b(cmp_b), a(cmp_a) {}

  constexpr Color(Color32 c, std::uint8_t white = 255,
                  std::uint8_t opaque = 255)
      : r(float(c.r) / white), g(float(c.g) / white), b(float(c.b) / white),
        a(float(c.a) / opaque) {}

  constexpr Color premultiplied() const { return {r * a, g * a, b * a, a}; }

  constexpr Color unpremultiplied() const { return {r / a, g / a, b / a, a}; }

  float distance(Color const &other) const {
    auto rmean = (other.r + r) / 2;
    auto diff = other - (*this);
    return std::sqrt((2.0f + rmean) * diff.r * diff.r + 4.0f * diff.g * diff.g +
                     (3.0f - rmean) * diff.b * diff.b);
  }

  float hue_distance(Color const &other) const {
    return std::abs(other.hue() - hue());
  }

  float hue() const {
    float min = utils::min(r, utils::min(g, b)),
          max = utils::max(r, utils::max(g, b));
    auto h = (max == r) ? (g - b) / (max - min)
                        : (max == g) ? 2.0f + (b - r) / (max - min)
                                     : 4.0f + (r - g) / (max - min);
    return (h >= 0.0f) ? h : h + 360.0f;
  }

  constexpr Color over(Color const &other) const {
    return (premultiplied() + other.premultiplied() * (1.0f - a))
        .unpremultiplied();
  }

  Color32 color32(std::uint8_t white = 255, std::uint8_t opaque = 255) const {
    using utils::clamp;
    return {std::uint8_t(std::lround(clamp(r, 0.0f, 1.0f) * white)),
            std::uint8_t(std::lround(clamp(g, 0.0f, 1.0f) * white)),
            std::uint8_t(std::lround(clamp(b, 0.0f, 1.0f) * white)),
            std::uint8_t(std::lround(clamp(a, 0.0f, 1.0f) * opaque))};
  }

  Color red(float value) const { return {value, g, b, a}; }

  Color green(float value) const { return {r, value, b, a}; }

  Color blue(float value) const { return {r, g, value, a}; }

  Color alpha(float value) const { return {r, g, b, value}; }

  Color &operator+=(Color const &other) { return (*this) = (*this) + other; }

  Color &operator-=(Color const &other) { return (*this) = (*this) - other; }

  Color &operator*=(float n) { return (*this) = (*this) * n; }

  Color &operator/=(float n) { return (*this) = (*this) / n; }

  constexpr bool operator==(Color const &other) const {
    return r == other.r && g == other.g && b == other.b && a == other.a;
  }

  constexpr bool operator!=(Color const &other) const {
    return r != other.r || g != other.g || b != other.b || a != other.a;
  }

  float r, g, b, a;
};

inline constexpr Color operator+(Color const &lhs, Color const &rhs) {
  return {lhs.r + rhs.r, lhs.g + rhs.g, lhs.b + rhs.b, lhs.a + rhs.a};
}

inline constexpr Color operator-(Color const &lhs, Color const &rhs) {
  return {lhs.r - rhs.r, lhs.g - rhs.g, lhs.b - rhs.b, lhs.a - rhs.a};
}

inline constexpr Color operator*(Color const &lhs, float const &rhs) {
  return {lhs.r * rhs, lhs.g * rhs, lhs.b * rhs, lhs.a * rhs};
}

inline constexpr Color operator*(float const &lhs, Color const &rhs) {
  return {lhs * rhs.r, lhs * rhs.g, lhs * rhs.b, lhs * rhs.a};
}

inline constexpr Color operator/(Color const &lhs, float const &rhs) {
  return {lhs.r / rhs, lhs.g / rhs, lhs.b / rhs, lhs.a / rhs};
}

inline constexpr Color operator/(float const &lhs, Color const &rhs) {
  return {lhs / rhs.r, lhs / rhs.g, lhs / rhs.b, lhs / rhs.a};
}

} /* namespace plot */

// include/terminal.hpp

#if defined(__unix__) || defined(__linux__) ||                                 \
    (defined(__APPLE__) && defined(__MACH__))
#define PLOT_PLATFORM_POSIX
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace plot {

enum class TerminalMode {
  None,     // Color not supported
  Minimal,  // Attribute reset and bold
  Ansi,     // ANSI 8-color palette
  Ansi256,  // ANSI (xterm) 256 color mode
  Iso24bit, // ISO-8613-3 24-bit true-color mode
  Windows   // Windows console API
};

enum class TerminalOp {
  Over,    // Paint source over destination, mix cell colors
  ClipDst, // Erase destination cell where source is not empty
  ClipSrc  // Ignore source cell where destination is not empty
};

namespace ansi {
namespace detail {
using ansi_color = std::pair<int, bool>;
using ansi_palette_entry = std::pair<plot::Color, ansi_color>;

static constexpr ansi_palette_entry palette[16] = {
    {{0, 0, 0}, {0, false}},                             // Black
    {{170.0f / 255.0f, 0, 0}, {1, false}},               // Red
    {{0, 170.0f / 255.0f, 0}, {2, false}},               // Green
    {{170.0f / 255.0f, 85.0f / 255.0f, 0}, {3, false}},  // Brown
    {{0, 0, 170.0f / 255.0f}, {4, false}},               // Blue
    {{170.0f / 255.0f, 0, 170.0f / 255.0f}, {5, false}}, // Magenta
    {{0, 170.0f / 255.0f, 170.0f / 255.0f}, {6, false}}, // Cyan
    {{170.0f / 255.0f, 170.0f / 255.0f, 170.0f / 255.0f}, {7, false}}, // Gray
    {{85.0f / 255.0f, 85.0f / 255.0f, 85.0f / 255.0f}, {0, true}}, // Darkgray
    {{1.0f, 85.0f / 255.0f, 85.0f / 255.0f}, {1, true}},           // Bright Red
    {{85.0f / 255.0f, 1.0f, 85.0f / 255.0f}, {2, true}}, // Bright green
    {{1.0f, 1.0f, 85.0f / 255.0f}, {3, true}},           // Yellow
    {{85.0f / 255.0f, 85.0f / 255.0f, 1.0f}, {4, true}}, // Bright Blue
    {{1.0f, 85.0f / 255.0f, 1.0f}, {5, true}},           // Bright Magenta
    {{85.0f / 255.0f, 1.0f, 1.0f}, {6, true}},           // Bright Cyan
    {{1.0f, 1.0f, 1.0f}, {7, true}}                      // White
};

inline ansi_palette_entry find_palette_entry(plot::Color c) {
  return *std::min_element(std::begin(palette), std::end(palette),
                           [c](auto const &e1, auto const &e2) {
                             return e1.first.distance(c) < e2.first.distance(c);
                           });
}

inline ansi_color find_color(plot::Color c) {
  return find_palette_entry(c).second;
}

inline std::uint8_t find_color256(plot::Color c) {
  using utils::clamp;
  auto ansi_clr = find_palette_entry(c);
  auto color = c.color32(5);
  std::uint8_t gray = std::lround(
      clamp(0.3f * c.r + 0.59f * c.g + 0.11f * c.b, 0.0f, 1.0f) * 23);

  auto ansi_dist = ansi_clr.first.distance(c);
  auto color_dist = plot::Color(color, 5).distance(c);
  auto gray_dist = plot::Color({gray, gray, gray, 255}, 23).distance(c);

  if (color_dist <= gray_dist && color_dist <= ansi_dist) {
    return 16 + 36 * color.r + 6 * color.g + color.b;
  } else if (gray_dist <= ansi_dist) {
    return gray + 0xe8;
  } else {
    return ansi_clr.second.first + 8 * ansi_clr.second.second;
  }
}

struct title_setter {
  string_view title;
};

inline std::ostream &operator<<(std::ostream &stream,
                                title_setter const &setter) {
  return stream << u8"\x1b]0;" << setter.title << u8"\x1b\\";
}

struct foreground_setter {
  ansi_color color;
};

inline std::ostream &operator<<(std::ostream &stream,
                                foreground_setter const &setter) {
  return stream << u8"\x1b[" << (30 + setter.color.first) << 'm';
}

struct background_setter {
  ansi_color color;
};

inline std::ostream &operator<<(std::ostream &stream,
                                background_setter const &setter) {
  return stream << u8"\x1b[" << (40 + setter.color.first) << 'm';
}

struct foreground_setter_256 {
  std::uint8_t code;
};

inline std::ostream &operator<<(std::ostream &stream,
                                foreground_setter_256 const &setter) {
  return stream << u8"\x1b[38;5;" << unsigned(setter.code) << 'm';
}

struct background_setter_256 {
  std::uint8_t code;
};

inline std::ostream &operator<<(std::ostream &stream,
                                background_setter_256 const &setter) {
  return stream << u8"\x1b[48;5;" << unsigned(setter.code) << 'm';
}

struct foreground_setter_24bit {
  Color32 color;
};

inline std::ostream &operator<<(std::ostream &stream,
                                foreground_setter_24bit const &setter) {
  return stream << u8"\x1b[38;2;" << unsigned(setter.color.r) << ';'
                << unsigned(setter.color.g) << ';' << unsigned(setter.color.b)
                << 'm';
}

struct background_setter_24bit {
  Color32 color;
};

inline std::ostream &operator<<(std::ostream &stream,
                                background_setter_24bit const &setter) {
  return stream << u8"\x1b[48;2;" << unsigned(setter.color.r) << ';'
                << unsigned(setter.color.g) << ';' << unsigned(setter.color.b)
                << 'm';
}

struct cursor_setter {
  Point loc;
};

inline std::ostream &operator<<(std::ostream &stream,
                                cursor_setter const &setter) {
  return stream << u8"\x1b[" << setter.loc.y << ';' << setter.loc.x << 'H';
}

enum class cursor_direction { up, down, forward, backward };

struct cursor_move {
  cursor_direction direction;
  unsigned count;
};

inline std::ostream &operator<<(std::ostream &stream, cursor_move const &move) {
  stream << u8"\x1b[" << move.count;

  switch (move.direction) {
  case cursor_direction::up:
    return stream << 'A';
  case cursor_direction::down:
    return stream << 'B';
  case cursor_direction::forward:
    return stream << 'C';
  case cursor_direction::backward:
    return stream << 'D';
  }

  return stream;
}
} /* namespace detail */

enum class Color {
  Black = 0,
  Red = 1,
  Green = 2,
  Brown = 3,
  Blue = 4,
  Magenta = 5,
  Cyan = 6,
  Gray = 7
};

inline detail::title_setter title(string_view title) { return {title}; }

inline std::ostream &reset(std::ostream &stream) {
  return stream << u8"\x1b[0m";
}

inline std::ostream &bold(std::ostream &stream) {
  return stream << u8"\x1b[1m";
}

inline std::ostream &clear(std::ostream &stream) {
  return stream << u8"\x1b[0;0H\x1b[2J";
}

inline std::ostream &clear_line(std::ostream &stream) {
  return stream << u8"\x1b[K";
}

inline std::ostream &line_start(std::ostream &stream) { return stream << '\r'; }

inline detail::foreground_setter foreground(Color c) {
  return {{int(c), false}};
}

inline detail::background_setter background(Color c) {
  return {{int(c), false}};
}

inline detail::foreground_setter foreground(plot::Color c) {
  return {detail::find_color(c)};
}

inline detail::background_setter background(plot::Color c) {
  return {detail::find_color(c)};
}

inline detail::foreground_setter_256 foreground256(plot::Color c) {
  return {detail::find_color256(c)};
}

inline detail::background_setter_256 background256(plot::Color c) {
  return {detail::find_color256(c)};
}

inline detail::foreground_setter_24bit foreground24bit(plot::Color c) {
  return {c.color32()};
}

inline detail::background_setter_24bit background24bit(plot::Color c) {
  return {c.color32()};
}

inline detail::cursor_setter move_to(Point loc) { return {loc}; }

inline detail::cursor_move move_up(unsigned count = 1) {
  return {detail::cursor_direction::up, count};
}

inline detail::cursor_move move_down(unsigned count = 1) {
  return {detail::cursor_direction::down, count};
}

inline detail::cursor_move move_forward(unsigned count = 1) {
  return {detail::cursor_direction::forward, count};
}

inline detail::cursor_move move_backward(unsigned count = 1) {
  return {detail::cursor_direction::backward, count};
}
} /* namespace ansi */

#ifdef PLOT_PLATFORM_POSIX

namespace detail {
template <typename T>
struct ansi_manip_wrapper {
  TerminalMode mode;
  T manip;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &stream,
                                ansi_manip_wrapper<T> const &wrapper) {
  if (wrapper.mode != TerminalMode::None &&
      wrapper.mode != TerminalMode::Windows)
    stream << wrapper.manip;

  return stream;
}

template <typename T>
inline ansi_manip_wrapper<std::decay_t<T>>
make_ansi_manip_wrapper(TerminalMode mode, T &&manip) {
  return {mode, std::forward<T>(manip)};
}

struct foreground_setter {
  TerminalMode mode;
  Color color;
};

inline std::ostream &operator<<(std::ostream &stream,
                                foreground_setter const &setter) {
  switch (setter.mode) {
  case TerminalMode::Ansi:
    return stream << ansi::foreground(setter.color);
  case TerminalMode::Ansi256:
    return stream << ansi::foreground256(setter.color);
  case TerminalMode::Iso24bit:
    return stream << ansi::foreground24bit(setter.color);
  default:
    return stream;
  }
}

struct background_setter {
  TerminalMode mode;
  Color color;
};

inline std::ostream &operator<<(std::ostream &stream,
                                background_setter const &setter) {
  switch (setter.mode) {
  case TerminalMode::Ansi:
    return stream << ansi::background(setter.color);
  case TerminalMode::Ansi256:
    return stream << ansi::background256(setter.color);
  case TerminalMode::Iso24bit:
    return stream << ansi::background24bit(setter.color);
  default:
    return stream;
  }
}
} /* namespace detail */

using Terminal = int;

class TerminalInfo {
public:
  explicit TerminalInfo(Terminal term = STDOUT_FILENO,
                        TerminalMode tmode = TerminalMode::None,
                        Color fgcolor = {0.9f, 0.9f, 0.9f, 1},
                        Color bgcolor = {0, 0, 0, 1})
      : mode(tmode), foreground_color(fgcolor), background_color(bgcolor),
        term_(term) {}

  bool is_terminal() const { return isatty(term_); }

  bool supported(TerminalMode m) const { return int(m) <= int(mode); }

  Size size() const {
    if (!is_terminal())
      return {};

    struct winsize ws;

    if (ioctl(term_, TIOCGWINSZ, &ws))
      return {};

    return {ws.ws_col, ws.ws_row};
  }

  // Query cursor position. Returns { 0, 0 } when not supported.
  //
  // XXX: This will discard all pending input data and sleep for 100ms
  // XXX: to wait for a response. It is the caller's responsibility to avoid
  // XXX: negative impact on users.
  //
  // XXX: This function is not thread-safe
  Point cursor() {
    Point loc;

    if (!is_terminal())
      return loc;

    auto response = query(u8"\x1b[6n", u8"R");
    if (!response.empty())
      std::sscanf(response.c_str(), u8"\x1b[%ld;%ldR", &loc.y, &loc.x);

    return loc;
  }

  // Detect terminal capabilities by inspecting the TERM and COLORTERM
  // environment variables. The mode property will be set only when
  // its current value is TerminalMode::None (the default).
  // If COLORTERM == "truecolor", assume 24-bit colors are supported.
  // If the terminal is compatible with xterm and the foreground_color
  // and background_color properties are set to white and black (the default),
  // query actual values by OSC 10 ; ? BEL and OSC 11 ; ? BEL
  //
  // XXX: This will discard all pending input data and sleep for 100ms
  // XXX: to wait for a response. It is the caller's responsibility to avoid
  // XXX: negative impact on users.
  //
  // XXX: This function is not thread-safe
  template <typename = void>
  TerminalInfo &detect();

  // Common control sequences
  // The following methods return IO manipulators for std::ostream

  auto title(string_view t) const {
    return detail::make_ansi_manip_wrapper(mode, ansi::title(t));
  }

  auto reset() const {
    return detail::make_ansi_manip_wrapper(mode, ansi::reset);
  }

  auto bold() const {
    return detail::make_ansi_manip_wrapper(mode, ansi::bold);
  }

  auto clear() const {
    return detail::make_ansi_manip_wrapper(mode, ansi::clear);
  }

  auto clear_line() const {
    return detail::make_ansi_manip_wrapper(mode, ansi::clear_line);
  }

  auto line_start() const {
    return detail::make_ansi_manip_wrapper(mode, ansi::line_start);
  }

  auto foreground(ansi::Color c) const {
    return detail::make_ansi_manip_wrapper(
        supported(TerminalMode::Ansi) ? TerminalMode::Ansi : TerminalMode::None,
        ansi::foreground(c));
  }

  auto background(ansi::Color c) const {
    return detail::make_ansi_manip_wrapper(
        supported(TerminalMode::Ansi) ? TerminalMode::Ansi : TerminalMode::None,
        ansi::background(c));
  }

  detail::foreground_setter foreground(Color c) const { return {mode, c}; }

  detail::background_setter background(Color c) const { return {mode, c}; }

  auto move_to(Point loc) const {
    return detail::make_ansi_manip_wrapper(mode, ansi::move_to(loc));
  }

  auto move_up(unsigned count = 1) const {
    return detail::make_ansi_manip_wrapper(mode, ansi::move_up(count));
  }

  auto move_down(unsigned count = 1) const {
    return detail::make_ansi_manip_wrapper(mode, ansi::move_down(count));
  }

  auto move_forward(unsigned count = 1) const {
    return detail::make_ansi_manip_wrapper(mode, ansi::move_forward(count));
  }

  auto move_backward(unsigned count = 1) const {
    return detail::make_ansi_manip_wrapper(mode, ansi::move_backward(count));
  }

  TerminalMode mode;
  Color foreground_color;
  Color background_color;

private:
  template <typename = void>
  std::string query(string_view cmd, string_view terminator);

  Terminal term_;
};

namespace detail {
struct tcsetattr_guard {
  tcsetattr_guard(Terminal t, struct termios oldtermios,
                  struct termios newtermios)
      : term(t), old(oldtermios), new_(newtermios) {}

  ~tcsetattr_guard() {
    if (ok)
      tcsetattr(term, TCSANOW, &old);
  }

  bool set() { return (ok = !tcsetattr(term, TCSANOW, &new_)); }

  bool ok = false;
  Terminal term;
  struct termios old, new_;
};
} /* namespace detail */

template <typename>
TerminalInfo &TerminalInfo::detect() {
  if (!is_terminal())
    return *this;

  std::string_view name, colorterm, vte_version;

  auto tmp = std::getenv(u8"TERM");
  if (tmp)
    name = tmp;

  tmp = std::getenv(u8"COLORTERM");
  if (tmp)
    colorterm = tmp;

  tmp = std::getenv(u8"VTE_VERSION");
  if (tmp)
    vte_version = tmp;

  bool xterm_like = detail::contains(name, u8"xterm");

  bool has_truecolor =
      !vte_version.empty()
          ? (vte_version[0] > '3' ||
             (vte_version[0] == '3' &&
              vte_version[1] >= '6')) // VTE >= 0.36 supports true color
          : detail::contains(colorterm, u8"truecolor") ||
                detail::contains(name, u8"cygwin") ||
                detail::contains(colorterm, u8"cygwin") ||
                detail::contains(name, u8"konsole") ||
                detail::contains(colorterm, u8"konsole");

  bool has_256color =
      has_truecolor || detail::contains(name, u8"256") || !colorterm.empty();

  bool has_ansi =
      has_256color || xterm_like || detail::contains(name, u8"screen") ||
      detail::contains(name, u8"vt100") || detail::contains(name, u8"color") ||
      detail::contains(name, u8"ansi") || detail::contains(name, u8"cygwin") ||
      detail::contains(name, u8"linux");

  if (mode == TerminalMode::None)
    mode = has_truecolor ? TerminalMode::Iso24bit
                         : has_256color ? TerminalMode::Ansi256
                                        : has_ansi ? TerminalMode::Ansi
                                                   : TerminalMode::None;

  if (xterm_like && foreground_color == Color(0.9f, 0.9f, 0.9f, 1)) {
    auto response = query(u8"\x1b]10;?\x1b\\", u8"\a\\");
    if (!response.empty()) {
      auto pos = response.find(u8"rgb:");
      if (pos != std::string::npos) {
        pos += 4;
        Color32 c = {255, 255, 255, 255};
        if (sscanf(response.c_str() + pos, u8"%2hhx%*2x/%2hhx%*2x/%2hhx%*2x",
                   &c.r, &c.g, &c.b) == 3)
          foreground_color = c;
      }
    }
  }

  if (xterm_like && background_color == Color(0, 0, 0, 1)) {
    auto response = query(u8"\x1b]11;?\x1b\\", u8"\a\\");
    if (!response.empty()) {
      auto pos = response.find(u8"rgb:");
      if (pos != std::string::npos) {
        pos += 4;
        Color32 c = {0, 0, 0, 255};
        if (sscanf(response.c_str() + pos, u8"%2hhx%*2x/%2hhx%*2x/%2hhx%*2x",
                   &c.r, &c.g, &c.b) == 3)
          background_color = c;
      }
    }
  }

  return *this;
}

template <typename>
std::string TerminalInfo::query(std::string_view cmd,
                                std::string_view terminator) {
  struct termios oldAttrs;
  if (tcgetattr(term_, &oldAttrs))
    return std::string();

  struct termios newAttrs = oldAttrs;
  newAttrs.c_lflag &= ~(ECHO | ICANON);
  newAttrs.c_cc[VMIN] = 0;
  newAttrs.c_cc[VTIME] = 0;

  detail::tcsetattr_guard guard(term_, oldAttrs, newAttrs);
  if (!guard.set())
    return std::string();

  if (tcdrain(term_))
    return std::string();

  if (tcflush(term_, TCIFLUSH))
    return std::string();

  if (std::size_t(write(term_, cmd.data(), cmd.size())) != cmd.size())
    return std::string();

  // FIXME: This won't be enough for remote terminals (e.g. SSH)
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(100ms);

  int available = 0;
  if (ioctl(term_, FIONREAD, &available))
    return std::string();

  std::string result;

  while (available) {
    result.reserve(result.size() + available);

    for (; available > 0; --available) {
      char ch = '\0';
      if (read(term_, &ch, 1) != 1)
        return std::string();

      if (!result.empty() || ch == '\x1b') {
        result.append(1, ch);
        if (terminator.find(ch) != std::string_view::npos)
          return result;
      }
    }

    // If we found an escape character but no terminator, continue reading
    if (!result.empty())
      if (ioctl(term_, FIONREAD, &available))
        return std::string();
  }

  return std::string();
}

#else
#error "Non-POSIX systems are not supported"
#endif

} /* namespace plot */

// include/colors.hpp

// http://prideout.net/archive/colors.php

namespace plot {

namespace palette {
constexpr Color aliceblue{0.941f, 0.973f, 1.000f};
constexpr Color antiquewhite{0.980f, 0.922f, 0.843f};
constexpr Color aqua{0.000f, 1.000f, 1.000f};
constexpr Color aquamarine{0.498f, 1.000f, 0.831f};
constexpr Color azure{0.941f, 1.000f, 1.000f};
constexpr Color beige{0.961f, 0.961f, 0.863f};
constexpr Color bisque{1.000f, 0.894f, 0.769f};
constexpr Color black{0.000f, 0.000f, 0.000f};
constexpr Color blanchedalmond{1.000f, 0.922f, 0.804f};
constexpr Color blue{0.000f, 0.000f, 1.000f};
constexpr Color blueviolet{0.541f, 0.169f, 0.886f};
constexpr Color brown{0.647f, 0.165f, 0.165f};
constexpr Color burlywood{0.871f, 0.722f, 0.529f};
constexpr Color cadetblue{0.373f, 0.620f, 0.627f};
constexpr Color chartreuse{0.498f, 1.000f, 0.000f};
constexpr Color chocolate{0.824f, 0.412f, 0.118f};
constexpr Color coral{1.000f, 0.498f, 0.314f};
constexpr Color cornflowerblue{0.392f, 0.584f, 0.929f};
constexpr Color cornsilk{1.000f, 0.973f, 0.863f};
constexpr Color crimson{0.863f, 0.078f, 0.235f};
constexpr Color cyan{0.000f, 1.000f, 1.000f};
constexpr Color darkblue{0.000f, 0.000f, 0.545f};
constexpr Color darkcyan{0.000f, 0.545f, 0.545f};
constexpr Color darkgoldenrod{0.722f, 0.525f, 0.043f};
constexpr Color darkgray{0.663f, 0.663f, 0.663f};
constexpr Color darkgreen{0.000f, 0.392f, 0.000f};
constexpr Color darkgrey{0.663f, 0.663f, 0.663f};
constexpr Color darkkhaki{0.741f, 0.718f, 0.420f};
constexpr Color darkmagenta{0.545f, 0.000f, 0.545f};
constexpr Color darkolivegreen{0.333f, 0.420f, 0.184f};
constexpr Color darkorange{1.000f, 0.549f, 0.000f};
constexpr Color darkorchid{0.600f, 0.196f, 0.800f};
constexpr Color darkred{0.545f, 0.000f, 0.000f};
constexpr Color darksalmon{0.914f, 0.588f, 0.478f};
constexpr Color darkseagreen{0.561f, 0.737f, 0.561f};
constexpr Color darkslateblue{0.282f, 0.239f, 0.545f};
constexpr Color darkslategray{0.184f, 0.310f, 0.310f};
constexpr Color darkslategrey{0.184f, 0.310f, 0.310f};
constexpr Color darkturquoise{0.000f, 0.808f, 0.820f};
constexpr Color darkviolet{0.580f, 0.000f, 0.827f};
constexpr Color deeppink{1.000f, 0.078f, 0.576f};
constexpr Color deepskyblue{0.000f, 0.749f, 1.000f};
constexpr Color dimgray{0.412f, 0.412f, 0.412f};
constexpr Color dimgrey{0.412f, 0.412f, 0.412f};
constexpr Color dodgerblue{0.118f, 0.565f, 1.000f};
constexpr Color firebrick{0.698f, 0.133f, 0.133f};
constexpr Color floralwhite{1.000f, 0.980f, 0.941f};
constexpr Color forestgreen{0.133f, 0.545f, 0.133f};
constexpr Color fuchsia{1.000f, 0.000f, 1.000f};
constexpr Color gainsboro{0.863f, 0.863f, 0.863f};
constexpr Color ghostwhite{0.973f, 0.973f, 1.000f};
constexpr Color gold{1.000f, 0.843f, 0.000f};
constexpr Color goldenrod{0.855f, 0.647f, 0.125f};
constexpr Color gray{0.502f, 0.502f, 0.502f};
constexpr Color green{0.000f, 0.502f, 0.000f};
constexpr Color greenyellow{0.678f, 1.000f, 0.184f};
constexpr Color grey{0.502f, 0.502f, 0.502f};
constexpr Color honeydew{0.941f, 1.000f, 0.941f};
constexpr Color hotpink{1.000f, 0.412f, 0.706f};
constexpr Color indianred{0.804f, 0.361f, 0.361f};
constexpr Color indigo{0.294f, 0.000f, 0.510f};
constexpr Color ivory{1.000f, 1.000f, 0.941f};
constexpr Color khaki{0.941f, 0.902f, 0.549f};
constexpr Color lavender{0.902f, 0.902f, 0.980f};
constexpr Color lavenderblush{1.000f, 0.941f, 0.961f};
constexpr Color lawngreen{0.486f, 0.988f, 0.000f};
constexpr Color lemonchiffon{1.000f, 0.980f, 0.804f};
constexpr Color lightblue{0.678f, 0.847f, 0.902f};
constexpr Color lightcoral{0.941f, 0.502f, 0.502f};
constexpr Color lightcyan{0.878f, 1.000f, 1.000f};
constexpr Color lightgoldenrodyellow{0.980f, 0.980f, 0.824f};
constexpr Color lightgray{0.827f, 0.827f, 0.827f};
constexpr Color lightgreen{0.565f, 0.933f, 0.565f};
constexpr Color lightgrey{0.827f, 0.827f, 0.827f};
constexpr Color lightpink{1.000f, 0.714f, 0.757f};
constexpr Color lightsalmon{1.000f, 0.627f, 0.478f};
constexpr Color lightseagreen{0.125f, 0.698f, 0.667f};
constexpr Color lightskyblue{0.529f, 0.808f, 0.980f};
constexpr Color lightslategray{0.467f, 0.533f, 0.600f};
constexpr Color lightslategrey{0.467f, 0.533f, 0.600f};
constexpr Color lightsteelblue{0.690f, 0.769f, 0.871f};
constexpr Color lightyellow{1.000f, 1.000f, 0.878f};
constexpr Color lime{0.000f, 1.000f, 0.000f};
constexpr Color limegreen{0.196f, 0.804f, 0.196f};
constexpr Color linen{0.980f, 0.941f, 0.902f};
constexpr Color magenta{1.000f, 0.000f, 1.000f};
constexpr Color maroon{0.502f, 0.000f, 0.000f};
constexpr Color mediumaquamarine{0.400f, 0.804f, 0.667f};
constexpr Color mediumblue{0.000f, 0.000f, 0.804f};
constexpr Color mediumorchid{0.729f, 0.333f, 0.827f};
constexpr Color mediumpurple{0.576f, 0.439f, 0.859f};
constexpr Color mediumseagreen{0.235f, 0.702f, 0.443f};
constexpr Color mediumslateblue{0.482f, 0.408f, 0.933f};
constexpr Color mediumspringgreen{0.000f, 0.980f, 0.604f};
constexpr Color mediumturquoise{0.282f, 0.820f, 0.800f};
constexpr Color mediumvioletred{0.780f, 0.082f, 0.522f};
constexpr Color midnightblue{0.098f, 0.098f, 0.439f};
constexpr Color mintcream{0.961f, 1.000f, 0.980f};
constexpr Color mistyrose{1.000f, 0.894f, 0.882f};
constexpr Color moccasin{1.000f, 0.894f, 0.710f};
constexpr Color navajowhite{1.000f, 0.871f, 0.678f};
constexpr Color navy{0.000f, 0.000f, 0.502f};
constexpr Color oldlace{0.992f, 0.961f, 0.902f};
constexpr Color olive{0.502f, 0.502f, 0.000f};
constexpr Color olivedrab{0.420f, 0.557f, 0.137f};
constexpr Color orange{1.000f, 0.647f, 0.000f};
constexpr Color orangered{1.000f, 0.271f, 0.000f};
constexpr Color orchid{0.855f, 0.439f, 0.839f};
constexpr Color palegoldenrod{0.933f, 0.910f, 0.667f};
constexpr Color palegreen{0.596f, 0.984f, 0.596f};
constexpr Color paleturquoise{0.686f, 0.933f, 0.933f};
constexpr Color palevioletred{0.859f, 0.439f, 0.576f};
constexpr Color papayawhip{1.000f, 0.937f, 0.835f};
constexpr Color peachpuff{1.000f, 0.855f, 0.725f};
constexpr Color peru{0.804f, 0.522f, 0.247f};
constexpr Color pink{1.000f, 0.753f, 0.796f};
constexpr Color plum{0.867f, 0.627f, 0.867f};
constexpr Color powderblue{0.690f, 0.878f, 0.902f};
constexpr Color purple{0.502f, 0.000f, 0.502f};
constexpr Color red{1.000f, 0.000f, 0.000f};
constexpr Color rosybrown{0.737f, 0.561f, 0.561f};
constexpr Color royalblue{0.255f, 0.412f, 0.882f};
constexpr Color saddlebrown{0.545f, 0.271f, 0.075f};
constexpr Color salmon{0.980f, 0.502f, 0.447f};
constexpr Color sandybrown{0.957f, 0.643f, 0.376f};
constexpr Color seagreen{0.180f, 0.545f, 0.341f};
constexpr Color seashell{1.000f, 0.961f, 0.933f};
constexpr Color sienna{0.627f, 0.322f, 0.176f};
constexpr Color silver{0.753f, 0.753f, 0.753f};
constexpr Color skyblue{0.529f, 0.808f, 0.922f};
constexpr Color slateblue{0.416f, 0.353f, 0.804f};
constexpr Color slategray{0.439f, 0.502f, 0.565f};
constexpr Color slategrey{0.439f, 0.502f, 0.565f};
constexpr Color snow{1.000f, 0.980f, 0.980f};
constexpr Color springgreen{0.000f, 1.000f, 0.498f};
constexpr Color steelblue{0.275f, 0.510f, 0.706f};
constexpr Color tan{0.824f, 0.706f, 0.549f};
constexpr Color teal{0.000f, 0.502f, 0.502f};
constexpr Color thistle{0.847f, 0.749f, 0.847f};
constexpr Color tomato{1.000f, 0.388f, 0.278f};
constexpr Color turquoise{0.251f, 0.878f, 0.816f};
constexpr Color violet{0.933f, 0.510f, 0.933f};
constexpr Color wheat{0.961f, 0.871f, 0.702f};
constexpr Color white{1.000f, 1.000f, 1.000f};
constexpr Color whitesmoke{0.961f, 0.961f, 0.961f};
constexpr Color yellow{1.000f, 1.000f, 0.000f};
constexpr Color yellowgreen{0.604f, 0.804f, 0.196f};
} /* namespace palette */

} /* namespace plot */

// include/rect.hpp

namespace plot {

template <typename T>
struct GenericRect;

template <typename T>
constexpr GenericRect<T> operator+(GenericRect<T> const &lhs,
                                   GenericPoint<T> const &rhs);

template <typename T>
constexpr GenericRect<T> operator-(GenericRect<T> const &lhs,
                                   GenericPoint<T> const &rhs);

template <typename T>
constexpr GenericRect<T> operator*(GenericRect<T> const &lhs, T const &rhs);

template <typename T>
constexpr GenericRect<T> operator*(T const &lhs, GenericRect<T> const &rhs);

template <typename T>
constexpr GenericRect<T> operator/(GenericRect<T> const &lhs, T const &rhs);

template <typename T>
constexpr GenericRect<T> operator/(T const &lhs, GenericRect<T> const &rhs);

template <typename T>
struct GenericRect {
  using coord_type = T;
  using point_type = GenericPoint<T>;

  constexpr GenericRect() = default;

  constexpr GenericRect(GenericPoint<T> const &point1,
                        GenericPoint<T> const &point2)
      : p1(point1), p2(point2) {}

  constexpr GenericRect(GenericSize<T> const &sz) : p1(), p2(sz) {}

  GenericRect sorted() const {
    auto x = utils::minmax(p1.x, p2.x);
    auto y = utils::minmax(p1.y, p2.y);
    return {{x.first, y.first}, {x.second, y.second}};
  }

  constexpr GenericRect sorted_x() const {
    return (p1.x > p2.x) ? GenericRect(p2, p1) : *this;
  }

  constexpr GenericRect sorted_y() const {
    return (p1.y > p2.y) ? GenericRect(p2, p1) : *this;
  }

  constexpr GenericPoint<T> size() const {
    return {utils::abs(p2.x - p1.x), utils::abs(p2.y - p1.y)};
  }

  // XXX: Calling on unsorted rectangles is undefined behavior
  constexpr bool contains(GenericPoint<T> p) const {
    return p.x >= p1.x && p.x < p2.x && p.y >= p1.y && p.y < p2.y;
  }

  // XXX: Calling on unsorted rectangles is undefined behavior
  constexpr bool contains(GenericRect const &r) const {
    return r.p1.x >= p1.x && r.p2.x <= p2.x && r.p1.y >= p1.y && r.p2.y <= p2.y;
  }

  constexpr GenericRect clamp(GenericRect const &r) const {
    return {p1.clamp(r.p1, r.p2), p2.clamp(r.p1, r.p2)};
  }

  template <typename U>
  constexpr operator GenericRect<U>() const {
    return {static_cast<GenericPoint<U>>(p1), static_cast<GenericPoint<U>>(p2)};
  }

  GenericRect &operator+=(GenericPoint<T> const &other) {
    return (*this) = (*this) + other;
  }

  GenericRect &operator-=(GenericPoint<T> const &other) {
    return (*this) = (*this) - other;
  }

  GenericRect &operator*=(T n) { return (*this) = (*this) * n; }

  GenericRect &operator/=(T n) { return (*this) = (*this) / n; }

  constexpr bool operator==(GenericRect const &other) const {
    return p1 == other.p1 && p2 == other.p2;
  }

  constexpr bool operator!=(GenericRect const &other) const {
    return p1 != other.p1 || p2 != other.p2;
  }

  GenericPoint<T> p1{}, p2{};
};

template <typename T>
inline constexpr GenericRect<T> operator+(GenericRect<T> const &lhs,
                                          GenericPoint<T> const &rhs) {
  return {lhs.p1 + rhs, lhs.p2 + rhs};
}

template <typename T>
inline constexpr GenericRect<T> operator-(GenericRect<T> const &lhs,
                                          GenericPoint<T> const &rhs) {
  return {lhs.p1 - rhs, lhs.p2 - rhs};
}

template <typename T>
inline constexpr GenericRect<T> operator*(GenericRect<T> const &lhs,
                                          T const &rhs) {
  return {lhs.p1 * rhs, lhs.p2 * rhs};
}

template <typename T>
inline constexpr GenericRect<T> operator*(T const &lhs,
                                          GenericRect<T> const &rhs) {
  return {lhs * rhs.p1, lhs * rhs.p2};
}

template <typename T>
inline constexpr GenericRect<T> operator/(GenericRect<T> const &lhs,
                                          T const &rhs) {
  return {lhs.p1 / rhs, lhs.p2 / rhs};
}

template <typename T>
inline constexpr GenericRect<T> operator/(T const &lhs,
                                          GenericRect<T> const &rhs) {
  return {lhs / rhs.p1, lhs / rhs.p2};
}

using Rect = GenericRect<Coord>;
using Rectf = GenericRect<Coordf>;

} /* namespace plot */

// include/unicode_structs.hpp

namespace plot {

namespace detail {
// Interval tree. Overlapping intervals are joined into a single one.
struct unicode_interval_tree_t {
  char32_t center;
  std::pair<char32_t, char32_t> interval;
  unicode_interval_tree_t const *left;
  unicode_interval_tree_t const *right;
};

template <typename = void>
bool unicode_cp_in_tree(char32_t cp, unicode_interval_tree_t const *tree) {
  while (tree) {
    if (cp >= tree->interval.first && cp <= tree->interval.second)
      return true;

    tree = (cp < tree->center) ? tree->left : tree->right;
  }

  return false;
}
} /* namespace detail */

} /* namespace plot */

// include/unicode_data.hpp

namespace plot {

namespace detail {
// Template variables struct members should be linked only once
// though appearing in multiple translation units
template <typename = void>
struct unicode_tables {
  static const unicode_interval_tree_t zero_width[];
  static const unicode_interval_tree_t double_width[];
};

template <typename T>
const unicode_interval_tree_t unicode_tables<T>::zero_width[] = {
    {0x1be8,
     {0x1be8, 0x1be9},
     &unicode_tables<T>::zero_width[1],
     &unicode_tables<T>::zero_width[152]},
    {0xccc,
     {0xccc, 0xccd},
     &unicode_tables<T>::zero_width[2],
     &unicode_tables<T>::zero_width[77]},
    {0x9bc,
     {0x9bc, 0x9bc},
     &unicode_tables<T>::zero_width[3],
     &unicode_tables<T>::zero_width[40]},
    {0x6eb,
     {0x6ea, 0x6ed},
     &unicode_tables<T>::zero_width[4],
     &unicode_tables<T>::zero_width[22]},
    {0x5c7,
     {0x5c7, 0x5c7},
     &unicode_tables<T>::zero_width[5],
     &unicode_tables<T>::zero_width[14]},
    {0x486,
     {0x483, 0x489},
     &unicode_tables<T>::zero_width[6],
     &unicode_tables<T>::zero_width[10]},
    {0x8f,
     {0x7f, 0x9f},
     &unicode_tables<T>::zero_width[7],
     &unicode_tables<T>::zero_width[8]},
    {0xf, {0x0, 0x1f}, nullptr, nullptr},
    {0xad, {0xad, 0xad}, nullptr, &unicode_tables<T>::zero_width[9]},
    {0x337, {0x300, 0x36f}, nullptr, nullptr},
    {0x5c1,
     {0x5c1, 0x5c2},
     &unicode_tables<T>::zero_width[11],
     &unicode_tables<T>::zero_width[13]},
    {0x5bf, {0x5bf, 0x5bf}, &unicode_tables<T>::zero_width[12], nullptr},
    {0x5a7, {0x591, 0x5bd}, nullptr, nullptr},
    {0x5c4, {0x5c4, 0x5c5}, nullptr, nullptr},
    {0x655,
     {0x64b, 0x65f},
     &unicode_tables<T>::zero_width[15],
     &unicode_tables<T>::zero_width[18]},
    {0x615,
     {0x610, 0x61a},
     &unicode_tables<T>::zero_width[16],
     &unicode_tables<T>::zero_width[17]},
    {0x602, {0x600, 0x605}, nullptr, nullptr},
    {0x61c, {0x61c, 0x61c}, nullptr, nullptr},
    {0x6e1,
     {0x6df, 0x6e4},
     &unicode_tables<T>::zero_width[19],
     &unicode_tables<T>::zero_width[21]},
    {0x6d9, {0x6d6, 0x6dd}, &unicode_tables<T>::zero_width[20], nullptr},
    {0x670, {0x670, 0x670}, nullptr, nullptr},
    {0x6e7, {0x6e7, 0x6e8}, nullptr, nullptr},
    {0x85a,
     {0x859, 0x85b},
     &unicode_tables<T>::zero_width[23],
     &unicode_tables<T>::zero_width[32]},
    {0x7ef,
     {0x7eb, 0x7f3},
     &unicode_tables<T>::zero_width[24],
     &unicode_tables<T>::zero_width[28]},
    {0x73d,
     {0x730, 0x74a},
     &unicode_tables<T>::zero_width[25],
     &unicode_tables<T>::zero_width[27]},
    {0x711, {0x711, 0x711}, &unicode_tables<T>::zero_width[26], nullptr},
    {0x70f, {0x70f, 0x70f}, nullptr, nullptr},
    {0x7ab, {0x7a6, 0x7b0}, nullptr, nullptr},
    {0x826,
     {0x825, 0x827},
     &unicode_tables<T>::zero_width[29],
     &unicode_tables<T>::zero_width[31]},
    {0x81f, {0x81b, 0x823}, &unicode_tables<T>::zero_width[30], nullptr},
    {0x817, {0x816, 0x819}, nullptr, nullptr},
    {0x82b, {0x829, 0x82d}, nullptr, nullptr},
    {0x94d,
     {0x94d, 0x94d},
     &unicode_tables<T>::zero_width[33],
     &unicode_tables<T>::zero_width[37]},
    {0x93a,
     {0x93a, 0x93a},
     &unicode_tables<T>::zero_width[34],
     &unicode_tables<T>::zero_width[35]},
    {0x8eb, {0x8d4, 0x902}, nullptr, nullptr},
    {0x944, {0x941, 0x948}, &unicode_tables<T>::zero_width[36], nullptr},
    {0x93c, {0x93c, 0x93c}, nullptr, nullptr},
    {0x962,
     {0x962, 0x963},
     &unicode_tables<T>::zero_width[38],
     &unicode_tables<T>::zero_width[39]},
    {0x954, {0x951, 0x957}, nullptr, nullptr},
    {0x981, {0x981, 0x981}, nullptr, nullptr},
    {0xb3c,
     {0xb3c, 0xb3c},
     &unicode_tables<T>::zero_width[41],
     &unicode_tables<T>::zero_width[59]},
    {0xa51,
     {0xa51, 0xa51},
     &unicode_tables<T>::zero_width[42],
     &unicode_tables<T>::zero_width[50]},
    {0xa3c,
     {0xa3c, 0xa3c},
     &unicode_tables<T>::zero_width[43],
     &unicode_tables<T>::zero_width[47]},
    {0x9cd,
     {0x9cd, 0x9cd},
     &unicode_tables<T>::zero_width[44],
     &unicode_tables<T>::zero_width[45]},
    {0x9c2, {0x9c1, 0x9c4}, nullptr, nullptr},
    {0xa01, {0xa01, 0xa02}, &unicode_tables<T>::zero_width[46], nullptr},
    {0x9e2, {0x9e2, 0x9e3}, nullptr, nullptr},
    {0xa47,
     {0xa47, 0xa48},
     &unicode_tables<T>::zero_width[48],
     &unicode_tables<T>::zero_width[49]},
    {0xa41, {0xa41, 0xa42}, nullptr, nullptr},
    {0xa4c, {0xa4b, 0xa4d}, nullptr, nullptr},
    {0xac3,
     {0xac1, 0xac5},
     &unicode_tables<T>::zero_width[51],
     &unicode_tables<T>::zero_width[55]},
    {0xa81,
     {0xa81, 0xa82},
     &unicode_tables<T>::zero_width[52],
     &unicode_tables<T>::zero_width[54]},
    {0xa70, {0xa70, 0xa71}, nullptr, &unicode_tables<T>::zero_width[53]},
    {0xa75, {0xa75, 0xa75}, nullptr, nullptr},
    {0xabc, {0xabc, 0xabc}, nullptr, nullptr},
    {0xae2,
     {0xae2, 0xae3},
     &unicode_tables<T>::zero_width[56],
     &unicode_tables<T>::zero_width[58]},
    {0xac7, {0xac7, 0xac8}, nullptr, &unicode_tables<T>::zero_width[57]},
    {0xacd, {0xacd, 0xacd}, nullptr, nullptr},
    {0xb01, {0xb01, 0xb01}, nullptr, nullptr},
    {0xc3f,
     {0xc3e, 0xc40},
     &unicode_tables<T>::zero_width[60],
     &unicode_tables<T>::zero_width[69]},
    {0xb62,
     {0xb62, 0xb63},
     &unicode_tables<T>::zero_width[61],
     &unicode_tables<T>::zero_width[65]},
    {0xb42,
     {0xb41, 0xb44},
     &unicode_tables<T>::zero_width[62],
     &unicode_tables<T>::zero_width[63]},
    {0xb3f, {0xb3f, 0xb3f}, nullptr, nullptr},
    {0xb4d, {0xb4d, 0xb4d}, nullptr, &unicode_tables<T>::zero_width[64]},
    {0xb56, {0xb56, 0xb56}, nullptr, nullptr},
    {0xbc0,
     {0xbc0, 0xbc0},
     &unicode_tables<T>::zero_width[66],
     &unicode_tables<T>::zero_width[67]},
    {0xb82, {0xb82, 0xb82}, nullptr, nullptr},
    {0xc00, {0xc00, 0xc00}, &unicode_tables<T>::zero_width[68], nullptr},
    {0xbcd, {0xbcd, 0xbcd}, nullptr, nullptr},
    {0xc62,
     {0xc62, 0xc63},
     &unicode_tables<T>::zero_width[70],
     &unicode_tables<T>::zero_width[73]},
    {0xc4b,
     {0xc4a, 0xc4d},
     &unicode_tables<T>::zero_width[71],
     &unicode_tables<T>::zero_width[72]},
    {0xc47, {0xc46, 0xc48}, nullptr, nullptr},
    {0xc55, {0xc55, 0xc56}, nullptr, nullptr},
    {0xcbc,
     {0xcbc, 0xcbc},
     &unicode_tables<T>::zero_width[74],
     &unicode_tables<T>::zero_width[75]},
    {0xc81, {0xc81, 0xc81}, nullptr, nullptr},
    {0xcc6, {0xcc6, 0xcc6}, &unicode_tables<T>::zero_width[76], nullptr},
    {0xcbf, {0xcbf, 0xcbf}, nullptr, nullptr},
    {0x135e,
     {0x135d, 0x135f},
     &unicode_tables<T>::zero_width[78],
     &unicode_tables<T>::zero_width[115]},
    {0xf39,
     {0xf39, 0xf39},
     &unicode_tables<T>::zero_width[79],
     &unicode_tables<T>::zero_width[97]},
    {0xe31,
     {0xe31, 0xe31},
     &unicode_tables<T>::zero_width[80],
     &unicode_tables<T>::zero_width[88]},
    {0xd4d,
     {0xd4d, 0xd4d},
     &unicode_tables<T>::zero_width[81],
     &unicode_tables<T>::zero_width[84]},
    {0xd01,
     {0xd01, 0xd01},
     &unicode_tables<T>::zero_width[82],
     &unicode_tables<T>::zero_width[83]},
    {0xce2, {0xce2, 0xce3}, nullptr, nullptr},
    {0xd42, {0xd41, 0xd44}, nullptr, nullptr},
    {0xdd3,
     {0xdd2, 0xdd4},
     &unicode_tables<T>::zero_width[85],
     &unicode_tables<T>::zero_width[87]},
    {0xd62, {0xd62, 0xd63}, nullptr, &unicode_tables<T>::zero_width[86]},
    {0xdca, {0xdca, 0xdca}, nullptr, nullptr},
    {0xdd6, {0xdd6, 0xdd6}, nullptr, nullptr},
    {0xebb,
     {0xebb, 0xebc},
     &unicode_tables<T>::zero_width[89],
     &unicode_tables<T>::zero_width[93]},
    {0xe4a,
     {0xe47, 0xe4e},
     &unicode_tables<T>::zero_width[90],
     &unicode_tables<T>::zero_width[91]},
    {0xe37, {0xe34, 0xe3a}, nullptr, nullptr},
    {0xeb1, {0xeb1, 0xeb1}, nullptr, &unicode_tables<T>::zero_width[92]},
    {0xeb6, {0xeb4, 0xeb9}, nullptr, nullptr},
    {0xf35,
     {0xf35, 0xf35},
     &unicode_tables<T>::zero_width[94],
     &unicode_tables<T>::zero_width[96]},
    {0xf18, {0xf18, 0xf19}, &unicode_tables<T>::zero_width[95], nullptr},
    {0xeca, {0xec8, 0xecd}, nullptr, nullptr},
    {0xf37, {0xf37, 0xf37}, nullptr, nullptr},
    {0x1039,
     {0x1039, 0x103a},
     &unicode_tables<T>::zero_width[98],
     &unicode_tables<T>::zero_width[106]},
    {0xf92,
     {0xf8d, 0xf97},
     &unicode_tables<T>::zero_width[99],
     &unicode_tables<T>::zero_width[102]},
    {0xf82,
     {0xf80, 0xf84},
     &unicode_tables<T>::zero_width[100],
     &unicode_tables<T>::zero_width[101]},
    {0xf77, {0xf71, 0xf7e}, nullptr, nullptr},
    {0xf86, {0xf86, 0xf87}, nullptr, nullptr},
    {0xfc6,
     {0xfc6, 0xfc6},
     &unicode_tables<T>::zero_width[103],
     &unicode_tables<T>::zero_width[104]},
    {0xfaa, {0xf99, 0xfbc}, nullptr, nullptr},
    {0x1034, {0x1032, 0x1037}, &unicode_tables<T>::zero_width[105], nullptr},
    {0x102e, {0x102d, 0x1030}, nullptr, nullptr},
    {0x1082,
     {0x1082, 0x1082},
     &unicode_tables<T>::zero_width[107],
     &unicode_tables<T>::zero_width[111]},
    {0x105f,
     {0x105e, 0x1060},
     &unicode_tables<T>::zero_width[108],
     &unicode_tables<T>::zero_width[110]},
    {0x1058, {0x1058, 0x1059}, &unicode_tables<T>::zero_width[109], nullptr},
    {0x103d, {0x103d, 0x103e}, nullptr, nullptr},
    {0x1072, {0x1071, 0x1074}, nullptr, nullptr},
    {0x108d,
     {0x108d, 0x108d},
     &unicode_tables<T>::zero_width[112],
     &unicode_tables<T>::zero_width[113]},
    {0x1085, {0x1085, 0x1086}, nullptr, nullptr},
    {0x109d, {0x109d, 0x109d}, nullptr, &unicode_tables<T>::zero_width[114]},
    {0x11af, {0x1160, 0x11ff}, nullptr, nullptr},
    {0x1a56,
     {0x1a56, 0x1a56},
     &unicode_tables<T>::zero_width[116],
     &unicode_tables<T>::zero_width[134]},
    {0x180c,
     {0x180b, 0x180e},
     &unicode_tables<T>::zero_width[117],
     &unicode_tables<T>::zero_width[126]},
    {0x17b4,
     {0x17b4, 0x17b5},
     &unicode_tables<T>::zero_width[118],
     &unicode_tables<T>::zero_width[122]},
    {0x1752,
     {0x1752, 0x1753},
     &unicode_tables<T>::zero_width[119],
     &unicode_tables<T>::zero_width[121]},
    {0x1733, {0x1732, 0x1734}, &unicode_tables<T>::zero_width[120], nullptr},
    {0x1713, {0x1712, 0x1714}, nullptr, nullptr},
    {0x1772, {0x1772, 0x1773}, nullptr, nullptr},
    {0x17ce,
     {0x17c9, 0x17d3},
     &unicode_tables<T>::zero_width[123],
     &unicode_tables<T>::zero_width[125]},
    {0x17c6, {0x17c6, 0x17c6}, &unicode_tables<T>::zero_width[124], nullptr},
    {0x17ba, {0x17b7, 0x17bd}, nullptr, nullptr},
    {0x17dd, {0x17dd, 0x17dd}, nullptr, nullptr},
    {0x1932,
     {0x1932, 0x1932},
     &unicode_tables<T>::zero_width[127],
     &unicode_tables<T>::zero_width[131]},
    {0x1921,
     {0x1920, 0x1922},
     &unicode_tables<T>::zero_width[128],
     &unicode_tables<T>::zero_width[130]},
    {0x1885, {0x1885, 0x1886}, nullptr, &unicode_tables<T>::zero_width[129]},
    {0x18a9, {0x18a9, 0x18a9}, nullptr, nullptr},
    {0x1927, {0x1927, 0x1928}, nullptr, nullptr},
    {0x1a17,
     {0x1a17, 0x1a18},
     &unicode_tables<T>::zero_width[132],
     &unicode_tables<T>::zero_width[133]},
    {0x193a, {0x1939, 0x193b}, nullptr, nullptr},
    {0x1a1b, {0x1a1b, 0x1a1b}, nullptr, nullptr},
    {0x1b38,
     {0x1b36, 0x1b3a},
     &unicode_tables<T>::zero_width[135],
     &unicode_tables<T>::zero_width[144]},
    {0x1a77,
     {0x1a73, 0x1a7c},
     &unicode_tables<T>::zero_width[136],
     &unicode_tables<T>::zero_width[140]},
    {0x1a60,
     {0x1a60, 0x1a60},
     &unicode_tables<T>::zero_width[137],
     &unicode_tables<T>::zero_width[138]},
    {0x1a5b, {0x1a58, 0x1a5e}, nullptr, nullptr},
    {0x1a68, {0x1a65, 0x1a6c}, &unicode_tables<T>::zero_width[139], nullptr},
    {0x1a62, {0x1a62, 0x1a62}, nullptr, nullptr},
    {0x1b01,
     {0x1b00, 0x1b03},
     &unicode_tables<T>::zero_width[141],
     &unicode_tables<T>::zero_width[143]},
    {0x1a7f, {0x1a7f, 0x1a7f}, nullptr, &unicode_tables<T>::zero_width[142]},
    {0x1ab7, {0x1ab0, 0x1abe}, nullptr, nullptr},
    {0x1b34, {0x1b34, 0x1b34}, nullptr, nullptr},
    {0x1ba3,
     {0x1ba2, 0x1ba5},
     &unicode_tables<T>::zero_width[145],
     &unicode_tables<T>::zero_width[149]},
    {0x1b42,
     {0x1b42, 0x1b42},
     &unicode_tables<T>::zero_width[146],
     &unicode_tables<T>::zero_width[147]},
    {0x1b3c, {0x1b3c, 0x1b3c}, nullptr, nullptr},
    {0x1b6f, {0x1b6b, 0x1b73}, nullptr, &unicode_tables<T>::zero_width[148]},
    {0x1b80, {0x1b80, 0x1b81}, nullptr, nullptr},
    {0x1bac,
     {0x1bab, 0x1bad},
     &unicode_tables<T>::zero_width[150],
     &unicode_tables<T>::zero_width[151]},
    {0x1ba8, {0x1ba8, 0x1ba9}, nullptr, nullptr},
    {0x1be6, {0x1be6, 0x1be6}, nullptr, nullptr},
    {0x11080,
     {0x1107f, 0x11081},
     &unicode_tables<T>::zero_width[153],
     &unicode_tables<T>::zero_width[228]},
    {0xa9bc,
     {0xa9bc, 0xa9bc},
     &unicode_tables<T>::zero_width[154],
     &unicode_tables<T>::zero_width[191]},
    {0x2d7f,
     {0x2d7f, 0x2d7f},
     &unicode_tables<T>::zero_width[155],
     &unicode_tables<T>::zero_width[173]},
    {0x1cf4,
     {0x1cf4, 0x1cf4},
     &unicode_tables<T>::zero_width[156],
     &unicode_tables<T>::zero_width[164]},
    {0x1c36,
     {0x1c36, 0x1c37},
     &unicode_tables<T>::zero_width[157],
     &unicode_tables<T>::zero_width[160]},
    {0x1bf0,
     {0x1bef, 0x1bf1},
     &unicode_tables<T>::zero_width[158],
     &unicode_tables<T>::zero_width[159]},
    {0x1bed, {0x1bed, 0x1bed}, nullptr, nullptr},
    {0x1c2f, {0x1c2c, 0x1c33}, nullptr, nullptr},
    {0x1cda,
     {0x1cd4, 0x1ce0},
     &unicode_tables<T>::zero_width[161],
     &unicode_tables<T>::zero_width[162]},
    {0x1cd1, {0x1cd0, 0x1cd2}, nullptr, nullptr},
    {0x1ced, {0x1ced, 0x1ced}, &unicode_tables<T>::zero_width[163], nullptr},
    {0x1ce5, {0x1ce2, 0x1ce8}, nullptr, nullptr},
    {0x202b,
     {0x2028, 0x202e},
     &unicode_tables<T>::zero_width[165],
     &unicode_tables<T>::zero_width[169]},
    {0x1dfd,
     {0x1dfb, 0x1dff},
     &unicode_tables<T>::zero_width[166],
     &unicode_tables<T>::zero_width[168]},
    {0x1dda, {0x1dc0, 0x1df5}, &unicode_tables<T>::zero_width[167], nullptr},
    {0x1cf8, {0x1cf8, 0x1cf9}, nullptr, nullptr},
    {0x200d, {0x200b, 0x200f}, nullptr, nullptr},
    {0x206a,
     {0x2066, 0x206f},
     &unicode_tables<T>::zero_width[170],
     &unicode_tables<T>::zero_width[171]},
    {0x2062, {0x2060, 0x2064}, nullptr, nullptr},
    {0x2cf0, {0x2cef, 0x2cf1}, &unicode_tables<T>::zero_width[172], nullptr},
    {0x20e0, {0x20d0, 0x20f0}, nullptr, nullptr},
    {0xa806,
     {0xa806, 0xa806},
     &unicode_tables<T>::zero_width[174],
     &unicode_tables<T>::zero_width[182]},
    {0xa670,
     {0xa66f, 0xa672},
     &unicode_tables<T>::zero_width[175],
     &unicode_tables<T>::zero_width[178]},
    {0x302b,
     {0x302a, 0x302d},
     &unicode_tables<T>::zero_width[176],
     &unicode_tables<T>::zero_width[177]},
    {0x2def, {0x2de0, 0x2dff}, nullptr, nullptr},
    {0x3099, {0x3099, 0x309a}, nullptr, nullptr},
    {0xa6f0,
     {0xa6f0, 0xa6f1},
     &unicode_tables<T>::zero_width[179],
     &unicode_tables<T>::zero_width[181]},
    {0xa69e, {0xa69e, 0xa69f}, &unicode_tables<T>::zero_width[180], nullptr},
    {0xa678, {0xa674, 0xa67d}, nullptr, nullptr},
    {0xa802, {0xa802, 0xa802}, nullptr, nullptr},
    {0xa929,
     {0xa926, 0xa92d},
     &unicode_tables<T>::zero_width[183],
     &unicode_tables<T>::zero_width[187]},
    {0xa8c4,
     {0xa8c4, 0xa8c5},
     &unicode_tables<T>::zero_width[184],
     &unicode_tables<T>::zero_width[186]},
    {0xa80b, {0xa80b, 0xa80b}, nullptr, &unicode_tables<T>::zero_width[185]},
    {0xa825, {0xa825, 0xa826}, nullptr, nullptr},
    {0xa8e8, {0xa8e0, 0xa8f1}, nullptr, nullptr},
    {0xa981,
     {0xa980, 0xa982},
     &unicode_tables<T>::zero_width[188],
     &unicode_tables<T>::zero_width[189]},
    {0xa94c, {0xa947, 0xa951}, nullptr, nullptr},
    {0xa9b3, {0xa9b3, 0xa9b3}, nullptr, &unicode_tables<T>::zero_width[190]},
    {0xa9b7, {0xa9b6, 0xa9b9}, nullptr, nullptr},
    {0xdb7f,
     {0xdb7f, 0xdb80},
     &unicode_tables<T>::zero_width[192],
     &unicode_tables<T>::zero_width[210]},
    {0xaab3,
     {0xaab2, 0xaab4},
     &unicode_tables<T>::zero_width[193],
     &unicode_tables<T>::zero_width[201]},
    {0xaa35,
     {0xaa35, 0xaa36},
     &unicode_tables<T>::zero_width[194],
     &unicode_tables<T>::zero_width[197]},
    {0xaa2b,
     {0xaa29, 0xaa2e},
     &unicode_tables<T>::zero_width[195],
     &unicode_tables<T>::zero_width[196]},
    {0xa9e5, {0xa9e5, 0xa9e5}, nullptr, nullptr},
    {0xaa31, {0xaa31, 0xaa32}, nullptr, nullptr},
    {0xaa7c,
     {0xaa7c, 0xaa7c},
     &unicode_tables<T>::zero_width[198],
     &unicode_tables<T>::zero_width[200]},
    {0xaa4c, {0xaa4c, 0xaa4c}, &unicode_tables<T>::zero_width[199], nullptr},
    {0xaa43, {0xaa43, 0xaa43}, nullptr, nullptr},
    {0xaab0, {0xaab0, 0xaab0}, nullptr, nullptr},
    {0xaaf6,
     {0xaaf6, 0xaaf6},
     &unicode_tables<T>::zero_width[202],
     &unicode_tables<T>::zero_width[206]},
    {0xaabe,
     {0xaabe, 0xaabf},
     &unicode_tables<T>::zero_width[203],
     &unicode_tables<T>::zero_width[204]},
    {0xaab7, {0xaab7, 0xaab8}, nullptr, nullptr},
    {0xaac1, {0xaac1, 0xaac1}, nullptr, &unicode_tables<T>::zero_width[205]},
    {0xaaec, {0xaaec, 0xaaed}, nullptr, nullptr},
    {0xabe8,
     {0xabe8, 0xabe8},
     &unicode_tables<T>::zero_width[207],
     &unicode_tables<T>::zero_width[208]},
    {0xabe5, {0xabe5, 0xabe5}, nullptr, nullptr},
    {0xabed, {0xabed, 0xabed}, nullptr, &unicode_tables<T>::zero_width[209]},
    {0xd800, {0xd800, 0xd800}, nullptr, nullptr},
    {0x102e0,
     {0x102e0, 0x102e0},
     &unicode_tables<T>::zero_width[211],
     &unicode_tables<T>::zero_width[219]},
    {0xfe27,
     {0xfe20, 0xfe2f},
     &unicode_tables<T>::zero_width[212],
     &unicode_tables<T>::zero_width[216]},
    {0xdfff,
     {0xdfff, 0xdfff},
     &unicode_tables<T>::zero_width[213],
     &unicode_tables<T>::zero_width[214]},
    {0xdbff, {0xdbff, 0xdc00}, nullptr, nullptr},
    {0xfb1e, {0xfb1e, 0xfb1e}, nullptr, &unicode_tables<T>::zero_width[215]},
    {0xfe07, {0xfe00, 0xfe0f}, nullptr, nullptr},
    {0xfffa,
     {0xfff9, 0xfffb},
     &unicode_tables<T>::zero_width[217],
     &unicode_tables<T>::zero_width[218]},
    {0xfeff, {0xfeff, 0xfeff}, nullptr, nullptr},
    {0x101fd, {0x101fd, 0x101fd}, nullptr, nullptr},
    {0x10a39,
     {0x10a38, 0x10a3a},
     &unicode_tables<T>::zero_width[220],
     &unicode_tables<T>::zero_width[224]},
    {0x10a02,
     {0x10a01, 0x10a03},
     &unicode_tables<T>::zero_width[221],
     &unicode_tables<T>::zero_width[222]},
    {0x10378, {0x10376, 0x1037a}, nullptr, nullptr},
    {0x10a0d, {0x10a0c, 0x10a0f}, &unicode_tables<T>::zero_width[223], nullptr},
    {0x10a05, {0x10a05, 0x10a06}, nullptr, nullptr},
    {0x11001,
     {0x11001, 0x11001},
     &unicode_tables<T>::zero_width[225],
     &unicode_tables<T>::zero_width[227]},
    {0x10a3f, {0x10a3f, 0x10a3f}, nullptr, &unicode_tables<T>::zero_width[226]},
    {0x10ae5, {0x10ae5, 0x10ae6}, nullptr, nullptr},
    {0x1103f, {0x11038, 0x11046}, nullptr, nullptr},
    {0x116b2,
     {0x116b0, 0x116b5},
     &unicode_tables<T>::zero_width[229],
     &unicode_tables<T>::zero_width[266]},
    {0x11340,
     {0x11340, 0x11340},
     &unicode_tables<T>::zero_width[230],
     &unicode_tables<T>::zero_width[248]},
    {0x111cb,
     {0x111ca, 0x111cc},
     &unicode_tables<T>::zero_width[231],
     &unicode_tables<T>::zero_width[240]},
    {0x11129,
     {0x11127, 0x1112b},
     &unicode_tables<T>::zero_width[232],
     &unicode_tables<T>::zero_width[236]},
    {0x110b9,
     {0x110b9, 0x110ba},
     &unicode_tables<T>::zero_width[233],
     &unicode_tables<T>::zero_width[234]},
    {0x110b4, {0x110b3, 0x110b6}, nullptr, nullptr},
    {0x11101, {0x11100, 0x11102}, &unicode_tables<T>::zero_width[235], nullptr},
    {0x110bd, {0x110bd, 0x110bd}, nullptr, nullptr},
    {0x11180,
     {0x11180, 0x11181},
     &unicode_tables<T>::zero_width[237],
     &unicode_tables<T>::zero_width[239]},
    {0x11130, {0x1112d, 0x11134}, nullptr, &unicode_tables<T>::zero_width[238]},
    {0x11173, {0x11173, 0x11173}, nullptr, nullptr},
    {0x111ba, {0x111b6, 0x111be}, nullptr, nullptr},
    {0x1123e,
     {0x1123e, 0x1123e},
     &unicode_tables<T>::zero_width[241],
     &unicode_tables<T>::zero_width[244]},
    {0x11234,
     {0x11234, 0x11234},
     &unicode_tables<T>::zero_width[242],
     &unicode_tables<T>::zero_width[243]},
    {0x11230, {0x1122f, 0x11231}, nullptr, nullptr},
    {0x11236, {0x11236, 0x11237}, nullptr, nullptr},
    {0x112e6,
     {0x112e3, 0x112ea},
     &unicode_tables<T>::zero_width[245],
     &unicode_tables<T>::zero_width[246]},
    {0x112df, {0x112df, 0x112df}, nullptr, nullptr},
    {0x11300, {0x11300, 0x11301}, nullptr, &unicode_tables<T>::zero_width[247]},
    {0x1133c, {0x1133c, 0x1133c}, nullptr, nullptr},
    {0x115b3,
     {0x115b2, 0x115b5},
     &unicode_tables<T>::zero_width[249],
     &unicode_tables<T>::zero_width[258]},
    {0x11446,
     {0x11446, 0x11446},
     &unicode_tables<T>::zero_width[250],
     &unicode_tables<T>::zero_width[254]},
    {0x1143b,
     {0x11438, 0x1143f},
     &unicode_tables<T>::zero_width[251],
     &unicode_tables<T>::zero_width[253]},
    {0x11369, {0x11366, 0x1136c}, nullptr, &unicode_tables<T>::zero_width[252]},
    {0x11372, {0x11370, 0x11374}, nullptr, nullptr},
    {0x11443, {0x11442, 0x11444}, nullptr, nullptr},
    {0x114ba,
     {0x114ba, 0x114ba},
     &unicode_tables<T>::zero_width[255],
     &unicode_tables<T>::zero_width[256]},
    {0x114b5, {0x114b3, 0x114b8}, nullptr, nullptr},
    {0x114c2, {0x114c2, 0x114c3}, &unicode_tables<T>::zero_width[257], nullptr},
    {0x114bf, {0x114bf, 0x114c0}, nullptr, nullptr},
    {0x1163d,
     {0x1163d, 0x1163d},
     &unicode_tables<T>::zero_width[259],
     &unicode_tables<T>::zero_width[263]},
    {0x115dc,
     {0x115dc, 0x115dd},
     &unicode_tables<T>::zero_width[260],
     &unicode_tables<T>::zero_width[262]},
    {0x115bc, {0x115bc, 0x115bd}, nullptr, &unicode_tables<T>::zero_width[261]},
    {0x115bf, {0x115bf, 0x115c0}, nullptr, nullptr},
    {0x11636, {0x11633, 0x1163a}, nullptr, nullptr},
    {0x116ab,
     {0x116ab, 0x116ab},
     &unicode_tables<T>::zero_width[264],
     &unicode_tables<T>::zero_width[265]},
    {0x1163f, {0x1163f, 0x11640}, nullptr, nullptr},
    {0x116ad, {0x116ad, 0x116ad}, nullptr, nullptr},
    {0x1d188,
     {0x1d185, 0x1d18b},
     &unicode_tables<T>::zero_width[267],
     &unicode_tables<T>::zero_width[285]},
    {0x11cb2,
     {0x11cb2, 0x11cb3},
     &unicode_tables<T>::zero_width[268],
     &unicode_tables<T>::zero_width[277]},
    {0x11c33,
     {0x11c30, 0x11c36},
     &unicode_tables<T>::zero_width[269],
     &unicode_tables<T>::zero_width[273]},
    {0x11723,
     {0x11722, 0x11725},
     &unicode_tables<T>::zero_width[270],
     &unicode_tables<T>::zero_width[272]},
    {0x1171e, {0x1171d, 0x1171f}, &unicode_tables<T>::zero_width[271], nullptr},
    {0x116b7, {0x116b7, 0x116b7}, nullptr, nullptr},
    {0x11729, {0x11727, 0x1172b}, nullptr, nullptr},
    {0x11c9c,
     {0x11c92, 0x11ca7},
     &unicode_tables<T>::zero_width[274],
     &unicode_tables<T>::zero_width[276]},
    {0x11c3a, {0x11c38, 0x11c3d}, nullptr, &unicode_tables<T>::zero_width[275]},
    {0x11c3f, {0x11c3f, 0x11c3f}, nullptr, nullptr},
    {0x11cad, {0x11caa, 0x11cb0}, nullptr, nullptr},
    {0x1bc9d,
     {0x1bc9d, 0x1bc9e},
     &unicode_tables<T>::zero_width[278],
     &unicode_tables<T>::zero_width[282]},
    {0x16af2,
     {0x16af0, 0x16af4},
     &unicode_tables<T>::zero_width[279],
     &unicode_tables<T>::zero_width[280]},
    {0x11cb5, {0x11cb5, 0x11cb6}, nullptr, nullptr},
    {0x16f90, {0x16f8f, 0x16f92}, &unicode_tables<T>::zero_width[281], nullptr},
    {0x16b33, {0x16b30, 0x16b36}, nullptr, nullptr},
    {0x1d168,
     {0x1d167, 0x1d169},
     &unicode_tables<T>::zero_width[283],
     &unicode_tables<T>::zero_width[284]},
    {0x1bca1, {0x1bca0, 0x1bca3}, nullptr, nullptr},
    {0x1d17a, {0x1d173, 0x1d182}, nullptr, nullptr},
    {0x1e003,
     {0x1e000, 0x1e006},
     &unicode_tables<T>::zero_width[286],
     &unicode_tables<T>::zero_width[294]},
    {0x1da75,
     {0x1da75, 0x1da75},
     &unicode_tables<T>::zero_width[287],
     &unicode_tables<T>::zero_width[291]},
    {0x1d243,
     {0x1d242, 0x1d244},
     &unicode_tables<T>::zero_width[288],
     &unicode_tables<T>::zero_width[289]},
    {0x1d1ab, {0x1d1aa, 0x1d1ad}, nullptr, nullptr},
    {0x1da53, {0x1da3b, 0x1da6c}, &unicode_tables<T>::zero_width[290], nullptr},
    {0x1da1b, {0x1da00, 0x1da36}, nullptr, nullptr},
    {0x1da9d,
     {0x1da9b, 0x1da9f},
     &unicode_tables<T>::zero_width[292],
     &unicode_tables<T>::zero_width[293]},
    {0x1da84, {0x1da84, 0x1da84}, nullptr, nullptr},
    {0x1daa8, {0x1daa1, 0x1daaf}, nullptr, nullptr},
    {0x1e8d3,
     {0x1e8d0, 0x1e8d6},
     &unicode_tables<T>::zero_width[295],
     &unicode_tables<T>::zero_width[299]},
    {0x1e01e,
     {0x1e01b, 0x1e021},
     &unicode_tables<T>::zero_width[296],
     &unicode_tables<T>::zero_width[297]},
    {0x1e010, {0x1e008, 0x1e018}, nullptr, nullptr},
    {0x1e028, {0x1e026, 0x1e02a}, &unicode_tables<T>::zero_width[298], nullptr},
    {0x1e023, {0x1e023, 0x1e024}, nullptr, nullptr},
    {0xe0001,
     {0xe0001, 0xe0001},
     &unicode_tables<T>::zero_width[300],
     &unicode_tables<T>::zero_width[301]},
    {0x1e947, {0x1e944, 0x1e94a}, nullptr, nullptr},
    {0xe004f, {0xe0020, 0xe007f}, nullptr, &unicode_tables<T>::zero_width[302]},
    {0xe0177, {0xe0100, 0xe01ef}, nullptr, nullptr},
};

template <typename T>
const unicode_interval_tree_t unicode_tables<T>::double_width[] = {
    {0xa96e,
     {0xa960, 0xa97c},
     &unicode_tables<T>::double_width[1],
     &unicode_tables<T>::double_width[53]},
    {0x274e,
     {0x274e, 0x274e},
     &unicode_tables<T>::double_width[2],
     &unicode_tables<T>::double_width[28]},
    {0x26bd,
     {0x26bd, 0x26be},
     &unicode_tables<T>::double_width[3],
     &unicode_tables<T>::double_width[16]},
    {0x25fd,
     {0x25fd, 0x25fe},
     &unicode_tables<T>::double_width[4],
     &unicode_tables<T>::double_width[10]},
    {0x23ea,
     {0x23e9, 0x23ec},
     &unicode_tables<T>::double_width[5],
     &unicode_tables<T>::double_width[8]},
    {0x231a,
     {0x231a, 0x231b},
     &unicode_tables<T>::double_width[6],
     &unicode_tables<T>::double_width[7]},
    {0x112f, {0x1100, 0x115f}, nullptr, nullptr},
    {0x2329, {0x2329, 0x232a}, nullptr, nullptr},
    {0x23f0, {0x23f0, 0x23f0}, nullptr, &unicode_tables<T>::double_width[9]},
    {0x23f3, {0x23f3, 0x23f3}, nullptr, nullptr},
    {0x2693,
     {0x2693, 0x2693},
     &unicode_tables<T>::double_width[11],
     &unicode_tables<T>::double_width[14]},
    {0x264d,
     {0x2648, 0x2653},
     &unicode_tables<T>::double_width[12],
     &unicode_tables<T>::double_width[13]},
    {0x2614, {0x2614, 0x2615}, nullptr, nullptr},
    {0x267f, {0x267f, 0x267f}, nullptr, nullptr},
    {0x26a1, {0x26a1, 0x26a1}, nullptr, &unicode_tables<T>::double_width[15]},
    {0x26aa, {0x26aa, 0x26ab}, nullptr, nullptr},
    {0x26fa,
     {0x26fa, 0x26fa},
     &unicode_tables<T>::double_width[17],
     &unicode_tables<T>::double_width[23]},
    {0x26d4,
     {0x26d4, 0x26d4},
     &unicode_tables<T>::double_width[18],
     &unicode_tables<T>::double_width[20]},
    {0x26c4, {0x26c4, 0x26c5}, nullptr, &unicode_tables<T>::double_width[19]},
    {0x26ce, {0x26ce, 0x26ce}, nullptr, nullptr},
    {0x26f2,
     {0x26f2, 0x26f3},
     &unicode_tables<T>::double_width[21],
     &unicode_tables<T>::double_width[22]},
    {0x26ea, {0x26ea, 0x26ea}, nullptr, nullptr},
    {0x26f5, {0x26f5, 0x26f5}, nullptr, nullptr},
    {0x270a,
     {0x270a, 0x270b},
     &unicode_tables<T>::double_width[24],
     &unicode_tables<T>::double_width[26]},
    {0x2705, {0x2705, 0x2705}, &unicode_tables<T>::double_width[25], nullptr},
    {0x26fd, {0x26fd, 0x26fd}, nullptr, nullptr},
    {0x2728, {0x2728, 0x2728}, nullptr, &unicode_tables<T>::double_width[27]},
    {0x274c, {0x274c, 0x274c}, nullptr, nullptr},
    {0x301f,
     {0x3000, 0x303e},
     &unicode_tables<T>::double_width[29],
     &unicode_tables<T>::double_width[41]},
    {0x2b1b,
     {0x2b1b, 0x2b1c},
     &unicode_tables<T>::double_width[30],
     &unicode_tables<T>::double_width[35]},
    {0x2796,
     {0x2795, 0x2797},
     &unicode_tables<T>::double_width[31],
     &unicode_tables<T>::double_width[33]},
    {0x2754, {0x2753, 0x2755}, nullptr, &unicode_tables<T>::double_width[32]},
    {0x2757, {0x2757, 0x2757}, nullptr, nullptr},
    {0x27bf, {0x27bf, 0x27bf}, &unicode_tables<T>::double_width[34], nullptr},
    {0x27b0, {0x27b0, 0x27b0}, nullptr, nullptr},
    {0x2e8c,
     {0x2e80, 0x2e99},
     &unicode_tables<T>::double_width[36],
     &unicode_tables<T>::double_width[38]},
    {0x2b55, {0x2b55, 0x2b55}, &unicode_tables<T>::double_width[37], nullptr},
    {0x2b50, {0x2b50, 0x2b50}, nullptr, nullptr},
    {0x2f6a,
     {0x2f00, 0x2fd5},
     &unicode_tables<T>::double_width[39],
     &unicode_tables<T>::double_width[40]},
    {0x2ec7, {0x2e9b, 0x2ef3}, nullptr, nullptr},
    {0x2ff5, {0x2ff0, 0x2ffb}, nullptr, nullptr},
    {0x31d1,
     {0x31c0, 0x31e3},
     &unicode_tables<T>::double_width[42],
     &unicode_tables<T>::double_width[47]},
    {0x3119,
     {0x3105, 0x312d},
     &unicode_tables<T>::double_width[43],
     &unicode_tables<T>::double_width[45]},
    {0x306b, {0x3041, 0x3096}, nullptr, &unicode_tables<T>::double_width[44]},
    {0x30cc, {0x3099, 0x30ff}, nullptr, nullptr},
    {0x31a5, {0x3190, 0x31ba}, &unicode_tables<T>::double_width[46], nullptr},
    {0x315f, {0x3131, 0x318e}, nullptr, nullptr},
    {0x405f,
     {0x3300, 0x4dbf},
     &unicode_tables<T>::double_width[48],
     &unicode_tables<T>::double_width[51]},
    {0x3233,
     {0x3220, 0x3247},
     &unicode_tables<T>::double_width[49],
     &unicode_tables<T>::double_width[50]},
    {0x3207, {0x31f0, 0x321e}, nullptr, nullptr},
    {0x32a7, {0x3250, 0x32fe}, nullptr, nullptr},
    {0xa4ab, {0xa490, 0xa4c6}, &unicode_tables<T>::double_width[52], nullptr},
    {0x7946, {0x4e00, 0xa48c}, nullptr, nullptr},
    {0x1f3e8,
     {0x1f3e0, 0x1f3f0},
     &unicode_tables<T>::double_width[54],
     &unicode_tables<T>::double_width[80]},
    {0x1f004,
     {0x1f004, 0x1f004},
     &unicode_tables<T>::double_width[55],
     &unicode_tables<T>::double_width[67]},
    {0xfe69,
     {0xfe68, 0xfe6b},
     &unicode_tables<T>::double_width[56],
     &unicode_tables<T>::double_width[61]},
    {0xfe14,
     {0xfe10, 0xfe19},
     &unicode_tables<T>::double_width[57],
     &unicode_tables<T>::double_width[59]},
    {0xc1d1, {0xac00, 0xd7a3}, nullptr, &unicode_tables<T>::double_width[58]},
    {0xf9ff, {0xf900, 0xfaff}, nullptr, nullptr},
    {0xfe5d, {0xfe54, 0xfe66}, &unicode_tables<T>::double_width[60], nullptr},
    {0xfe41, {0xfe30, 0xfe52}, nullptr, nullptr},
    {0x16fe0,
     {0x16fe0, 0x16fe0},
     &unicode_tables<T>::double_width[62],
     &unicode_tables<T>::double_width[64]},
    {0xff30, {0xff01, 0xff60}, nullptr, &unicode_tables<T>::double_width[63]},
    {0xffe3, {0xffe0, 0xffe6}, nullptr, nullptr},
    {0x18979,
     {0x18800, 0x18af2},
     &unicode_tables<T>::double_width[65],
     &unicode_tables<T>::double_width[66]},
    {0x17bf6, {0x17000, 0x187ec}, nullptr, nullptr},
    {0x1b000, {0x1b000, 0x1b001}, nullptr, nullptr},
    {0x1f250,
     {0x1f250, 0x1f251},
     &unicode_tables<T>::double_width[68],
     &unicode_tables<T>::double_width[74]},
    {0x1f201,
     {0x1f200, 0x1f202},
     &unicode_tables<T>::double_width[69],
     &unicode_tables<T>::double_width[72]},
    {0x1f18e,
     {0x1f18e, 0x1f18e},
     &unicode_tables<T>::double_width[70],
     &unicode_tables<T>::double_width[71]},
    {0x1f0cf, {0x1f0cf, 0x1f0cf}, nullptr, nullptr},
    {0x1f195, {0x1f191, 0x1f19a}, nullptr, nullptr},
    {0x1f244,
     {0x1f240, 0x1f248},
     &unicode_tables<T>::double_width[73],
     nullptr},
    {0x1f225, {0x1f210, 0x1f23b}, nullptr, nullptr},
    {0x1f388,
     {0x1f37e, 0x1f393},
     &unicode_tables<T>::double_width[75],
     &unicode_tables<T>::double_width[78]},
    {0x1f331,
     {0x1f32d, 0x1f335},
     &unicode_tables<T>::double_width[76],
     &unicode_tables<T>::double_width[77]},
    {0x1f310, {0x1f300, 0x1f320}, nullptr, nullptr},
    {0x1f359, {0x1f337, 0x1f37c}, nullptr, nullptr},
    {0x1f3d1,
     {0x1f3cf, 0x1f3d3},
     &unicode_tables<T>::double_width[79],
     nullptr},
    {0x1f3b5, {0x1f3a0, 0x1f3ca}, nullptr, nullptr},
    {0x1f6cc,
     {0x1f6cc, 0x1f6cc},
     &unicode_tables<T>::double_width[81],
     &unicode_tables<T>::double_width[93]},
    {0x1f55b,
     {0x1f550, 0x1f567},
     &unicode_tables<T>::double_width[82],
     &unicode_tables<T>::double_width[88]},
    {0x1f49f,
     {0x1f442, 0x1f4fc},
     &unicode_tables<T>::double_width[83],
     &unicode_tables<T>::double_width[86]},
    {0x1f41b,
     {0x1f3f8, 0x1f43e},
     &unicode_tables<T>::double_width[84],
     &unicode_tables<T>::double_width[85]},
    {0x1f3f4, {0x1f3f4, 0x1f3f4}, nullptr, nullptr},
    {0x1f440, {0x1f440, 0x1f440}, nullptr, nullptr},
    {0x1f51e,
     {0x1f4ff, 0x1f53d},
     nullptr,
     &unicode_tables<T>::double_width[87]},
    {0x1f54c, {0x1f54b, 0x1f54e}, nullptr, nullptr},
    {0x1f5a4,
     {0x1f5a4, 0x1f5a4},
     &unicode_tables<T>::double_width[89],
     &unicode_tables<T>::double_width[91]},
    {0x1f595,
     {0x1f595, 0x1f596},
     &unicode_tables<T>::double_width[90],
     nullptr},
    {0x1f57a, {0x1f57a, 0x1f57a}, nullptr, nullptr},
    {0x1f625,
     {0x1f5fb, 0x1f64f},
     nullptr,
     &unicode_tables<T>::double_width[92]},
    {0x1f6a2, {0x1f680, 0x1f6c5}, nullptr, nullptr},
    {0x1f938,
     {0x1f933, 0x1f93e},
     &unicode_tables<T>::double_width[94],
     &unicode_tables<T>::double_width[100]},
    {0x1f6f5,
     {0x1f6f4, 0x1f6f6},
     &unicode_tables<T>::double_width[95],
     &unicode_tables<T>::double_width[97]},
    {0x1f6eb,
     {0x1f6eb, 0x1f6ec},
     &unicode_tables<T>::double_width[96],
     nullptr},
    {0x1f6d1, {0x1f6d0, 0x1f6d2}, nullptr, nullptr},
    {0x1f923,
     {0x1f920, 0x1f927},
     &unicode_tables<T>::double_width[98],
     &unicode_tables<T>::double_width[99]},
    {0x1f917, {0x1f910, 0x1f91e}, nullptr, nullptr},
    {0x1f930, {0x1f930, 0x1f930}, nullptr, nullptr},
    {0x1f988,
     {0x1f980, 0x1f991},
     &unicode_tables<T>::double_width[101],
     &unicode_tables<T>::double_width[103]},
    {0x1f945,
     {0x1f940, 0x1f94b},
     nullptr,
     &unicode_tables<T>::double_width[102]},
    {0x1f957, {0x1f950, 0x1f95e}, nullptr, nullptr},
    {0x27ffe,
     {0x20000, 0x2fffd},
     &unicode_tables<T>::double_width[104],
     &unicode_tables<T>::double_width[105]},
    {0x1f9c0, {0x1f9c0, 0x1f9c0}, nullptr, nullptr},
    {0x37ffe, {0x30000, 0x3fffd}, nullptr, nullptr},
};
} /* namespace detail */

} /* namespace plot */

// include/unicode.hpp

namespace plot {

namespace detail {
// For code points in classes Cc, Cf, Cn, Cs, Me, Mn, Zl, Zp,
// and in range U+1160..U+11FF (Korean combining characters),
// return width 0.
//
// For code points with East_Asian_Width property set to F (Fullwidth)
// or W (Wide), return width 2.
//
// For all other code points, return 1.
inline std::size_t wcwidth(char32_t cp) {
  if (unicode_cp_in_tree(cp, unicode_tables<>::zero_width))
    return 0;

  if (unicode_cp_in_tree(cp, unicode_tables<>::double_width))
    return 2;

  return 1;
}

constexpr std::uint8_t utf8_start_masks[] = {0, 0b1111111, 0b11111, 0b1111,
                                             0b111};

constexpr std::uint8_t utf8_start_markers[] = {0, 0b00000000, 0b11000000,
                                               0b11100000, 0b11110000};

constexpr std::uint8_t utf8_cont_mask = 0b111111;
constexpr std::uint8_t utf8_cont_marker = 0b10000000;

inline constexpr bool utf8_seq_start(std::uint8_t byte) {
  return (byte & ~utf8_start_masks[1]) == utf8_start_markers[1] ||
         (byte & ~utf8_start_masks[2]) == utf8_start_markers[2] ||
         (byte & ~utf8_start_masks[3]) == utf8_start_markers[3] ||
         (byte & ~utf8_start_masks[4]) == utf8_start_markers[4];
}

inline constexpr bool utf8_seq_cont(std::uint8_t byte) {
  return (byte & ~utf8_cont_mask) == utf8_cont_marker;
}

inline int utf8_seq_length(std::uint8_t first) {
  if ((first & ~utf8_start_masks[1]) == utf8_start_markers[1])
    return 1;
  else if ((first & ~utf8_start_masks[2]) == utf8_start_markers[2])
    return 2;
  else if ((first & ~utf8_start_masks[3]) == utf8_start_markers[3])
    return 3;
  else if ((first & ~utf8_start_masks[4]) == utf8_start_markers[4])
    return 4;
  else
    return 0;
}

template <typename Iterator>
inline Iterator utf8_next(Iterator it, Iterator end) {
  while (it != end && !utf8_seq_start(*++it)) { /* do nothing */
  }
  return it;
}

template <typename Iterator>
inline char32_t utf8_cp(Iterator it, Iterator end) {
  char32_t cp = static_cast<std::uint8_t>(*it);
  auto len = utf8_seq_length(cp);

  if (!len)
    return std::numeric_limits<char32_t>::max();

  cp = (cp & utf8_start_masks[len]);

  while (--len > 0 && it != end && utf8_seq_cont(*++it))
    cp = (cp << 6) | (static_cast<std::uint8_t>(*it) & utf8_cont_mask);

  while (--len > 0)
    cp <<= 6;

  return cp;
}
} /* namespace detail */

template <typename Iterator>
std::size_t utf8_string_width(Iterator first, Iterator last) {
  std::size_t width = 0;

  for (; first != last; first = detail::utf8_next(first, last))
    width += detail::wcwidth(detail::utf8_cp(first, last));

  return width;
}

inline std::size_t utf8_string_width(std::string_view str) {
  return utf8_string_width(str.begin(), str.end());
}

template <typename Iterator>
std::pair<Iterator, std::size_t> utf8_clamp(Iterator first, Iterator last,
                                            std::size_t width) {
  std::size_t request = width;

  for (; first != last; first = detail::utf8_next(first, last)) {
    auto cw = detail::wcwidth(detail::utf8_cp(first, last));
    if (cw > width)
      break;
    width -= cw;
  }

  return {first, request - width};
}

inline std::pair<std::string_view, std::size_t> utf8_clamp(std::string_view str,
                                                           std::size_t width) {
  auto res = utf8_clamp(str.begin(), str.end(), width);
  return {std::string_view(str.begin(), res.first - str.begin()), res.second};
}

} /* namespace plot */

// include/layout.hpp

namespace plot {

namespace detail {
template <typename Block, typename Line>
class block_iterator {
public:
  using value_type = Line;
  using reference = value_type const &;
  using pointer = value_type const *;
  using difference_type = Coord;
  using iterator_category = std::forward_iterator_tag;

  block_iterator() = default;

  reference operator*() const { return line_; }

  pointer operator->() const { return &line_; }

  block_iterator &operator++() {
    line_ = line_.next();
    return *this;
  }

  block_iterator operator++(int) {
    block_iterator prev = *this;
    line_ = line_.next();
    return prev;
  }

  bool operator==(block_iterator const &other) const {
    return line_.equal(other.line_);
  }

  bool operator!=(block_iterator const &other) const {
    return !line_.equal(other.line_);
  }

private:
  friend Block;

  block_iterator(Line line) : line_(std::move(line)) {}

  Line line_;
};

template <typename Block>
class single_line_adapter {
public:
  using value_type = Block;
  using reference = value_type const &;
  using pointer = value_type const *;
  using difference_type = Coord;
  using iterator_category = std::forward_iterator_tag;

  single_line_adapter() = default;

  reference operator*() const { return *block_; }

  pointer operator->() const { return block_; }

  single_line_adapter &operator++() {
    end_ = true;
    return *this;
  }

  single_line_adapter operator++(int) {
    single_line_adapter prev = *this;
    end_ = true;
    return prev;
  }

  bool operator==(single_line_adapter const &other) const {
    return end_ == other.end_ && block_ == other.block_;
  }

  bool operator!=(single_line_adapter const &other) const {
    return end_ != other.end_ || block_ != other.block_;
  }

private:
  template <typename, bool>
  friend struct normal_block_ref_traits;

  single_line_adapter(pointer block, bool end = false)
      : block_(block), end_(end) {}

  pointer block_;
  bool end_;
};

template <typename Block>
inline single_line_adapter<Block>
operator+(typename single_line_adapter<Block>::difference_type n,
          single_line_adapter<Block> const &it) {
  return it + n;
}

template <typename T>
struct is_canvas {
  template <typename S>
  static constexpr bool test(decltype(std::declval<S>().char_size()) const *) {
    return true;
  }

  template <typename S>
  static constexpr bool test(...) {
    return false;
  }

  static constexpr bool value = test<T>(0);
};

template <typename Block,
          bool =
              std::is_same<Size, decltype(std::declval<Block>().size())>::value>
struct normal_block_ref_traits {
  using iterator = single_line_adapter<Block>;

  static Size size(Block const &block) { return {block.size(), 1}; }

  static iterator begin(Block const &block) { return {&block}; }

  static iterator end(Block const &block) { return {&block, true}; }
};

template <typename Block>
struct normal_block_ref_traits<Block, true> {
  using iterator = typename Block::const_iterator;

  static Size size(Block const &block) { return block.size(); }

  static iterator begin(Block const &block) { return std::begin(block); }

  static iterator end(Block const &block) { return std::end(block); }
};

template <typename Block, bool = is_canvas<Block>::value>
struct block_ref_traits : normal_block_ref_traits<Block> {};

template <typename Block>
struct block_ref_traits<Block, true> : normal_block_ref_traits<Block> {
  static Size size(Block const &block) { return block.char_size(); }
};

template <typename Block>
struct block_traits : block_ref_traits<Block> {};

template <typename Block>
struct block_traits<Block *> {
  using iterator = typename block_ref_traits<Block>::iterator;

  static auto size(Block *block) {
    return block_ref_traits<Block>::size(*block);
  }

  static auto begin(Block *block) {
    return block_ref_traits<Block>::begin(*block);
  }

  static auto end(Block *block) { return block_ref_traits<Block>::end(*block); }
};
} /* namespace detail */

enum class Align { Left, Center, Right };

enum class VAlign { Top, Middle, Bottom };

enum class BorderStyle {
  None,
  Solid,
  SolidBold,
  Dashed,
  DashedBold,
  Dotted,
  DottedBold,
  Double
};

struct Border {
  Border(BorderStyle style = BorderStyle::None, bool rounded_corners = false) {
    switch (style) {
    case BorderStyle::None:
      top_left = top = top_right = left = right = bottom_left = bottom =
          bottom_right = u8" ";
      return;
    case BorderStyle::Solid:
      top = bottom = u8"─";
      left = right = u8"│";
      break;
    case BorderStyle::SolidBold:
      top = bottom = u8"━";
      left = right = u8"┃";
      break;
    case BorderStyle::Dashed:
      top = u8"╴";
      bottom = u8"╶";
      left = u8"╷";
      right = u8"╵";
      break;
    case BorderStyle::DashedBold:
      top = u8"╸";
      bottom = u8"╺";
      left = u8"╻";
      right = u8"╹";
      break;
    case BorderStyle::Dotted:
      top = bottom = u8"┈";
      left = right = u8"┊";
      break;
    case BorderStyle::DottedBold:
      top = bottom = u8"┉";
      left = right = u8"┋";
      break;
    case BorderStyle::Double:
      top = bottom = u8"═";
      left = right = u8"║";
      top_left = u8"╔";
      top_right = u8"╗";
      bottom_left = u8"╚";
      bottom_right = u8"╝";
      return;
    }

    switch (style) {
    case BorderStyle::Solid:
    case BorderStyle::Dashed:
    case BorderStyle::Dotted:
      top_left = (rounded_corners) ? u8"╭" : u8"┌";
      top_right = (rounded_corners) ? u8"╮" : u8"┐";
      bottom_left = (rounded_corners) ? u8"╰" : u8"└";
      bottom_right = (rounded_corners) ? u8"╯" : u8"┘";
      return;
    case BorderStyle::SolidBold:
    case BorderStyle::DashedBold:
    case BorderStyle::DottedBold:
      top_left = u8"┏";
      top_right = u8"┓";
      bottom_left = u8"┗";
      bottom_right = u8"┛";
      return;
    default:
      return;
    }
  }

  std::string_view top_left, top, top_right, left, right, bottom_left, bottom,
      bottom_right;
};

class Label;

namespace detail {
class label_line;

std::ostream &operator<<(std::ostream &, label_line const &);

class label_line {
  friend class detail::block_iterator<Label, label_line>;
  friend class plot::Label;

  friend std::ostream &operator<<(std::ostream &, label_line const &);

  label_line(Label const *label, std::ptrdiff_t overflow)
      : label_(label), overflow_(overflow) {}

  label_line next() const { return label_line(label_, overflow_ + 1); }

  bool equal(label_line const &other) const {
    return overflow_ == other.overflow_;
  }

  Label const *label_ = nullptr;
  std::ptrdiff_t overflow_ = 0;

public:
  label_line() = default;
};
} /* namespace detail */

class Label {
public:
  using value_type = detail::label_line;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<Label, value_type>;
  using iterator = const_iterator;
  using difference_type = typename const_iterator::difference_type;
  using size_type = Size;

  explicit Label(std::string_view text, std::size_t width = 0,
                 std::string_view fill = " ")
      : text_(text), width_(width), fill_(fill) {}

  explicit Label(std::string_view text, Align align, std::size_t width = 0,
                 std::string_view fill = " ")
      : text_(text), align_(align), width_(width), fill_(fill) {}

  Size size() const {
    return {Coord(width_ ? width_ : utf8_string_width(text_)), 1};
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const { return {{this, 0}}; }

  const_iterator cend() const { return {{this, 1}}; }

private:
  friend std::ostream &detail::operator<<(std::ostream &, value_type const &);

  std::string_view text_;
  Align align_ = Align::Left;
  std::size_t width_ = 0;
  std::string_view fill_ = " ";
};

inline std::ostream &operator<<(std::ostream &stream, Label const &label) {
  auto line = *label.cbegin();
  stream << line << '\n';

  return stream;
}

namespace detail {
inline std::ostream &operator<<(std::ostream &stream, label_line const &line) {
  auto const text = line.label_->text_;
  auto const twidth = utf8_string_width(text);
  auto const width = line.label_->width_;

  if (!width || twidth == width)
    return stream << text;

  if (twidth > width)
    return stream << utf8_clamp(text, width).first;

  std::size_t padding = width - twidth;
  std::size_t padding_left =
      (line.label_->align_ == Align::Center)
          ? padding / 2
          : (line.label_->align_ == Align::Right) ? padding : 0;
  std::size_t padding_right = padding - padding_left;

  while (padding_left--)
    stream << line.label_->fill_;

  stream << text;

  while (padding_right--)
    stream << line.label_->fill_;

  return stream;
}
} /* namespace detail */

inline Label label(std::string_view text, std::size_t width = 0,
                   std::string_view fill = " ") {
  return Label(text, width, fill);
}

inline Label label(std::string_view text, Align align, std::size_t width = 0,
                   std::string_view fill = " ") {
  return Label(text, align, width, fill);
}

template <typename Block>
class Alignment;

namespace detail {
template <typename Block>
class alignment_line;

template <typename Block>
std::ostream &operator<<(std::ostream &, alignment_line<Block> const &);

template <typename Block>
class alignment_line {
  using block_iterator = typename detail::block_traits<Block>::iterator;

  friend class detail::block_iterator<Alignment<Block>, alignment_line>;
  friend class Alignment<Block>;

  friend std::ostream &operator<<<Block>(std::ostream &,
                                         alignment_line const &);

  alignment_line(Alignment<Block> const *alignment, std::size_t left_margin,
                 std::size_t right_margin, std::ptrdiff_t overflow,
                 block_iterator line, block_iterator end)
      : alignment_(alignment), left_margin_(left_margin),
        right_margin_(right_margin), overflow_(overflow),
        line_(std::move(line)), end_(std::move(end)) {}

  alignment_line next() const {
    return (overflow_ || line_ == end_)
               ? alignment_line(alignment_, left_margin_, right_margin_,
                                overflow_ + 1, line_, end_)
               : alignment_line(alignment_, left_margin_, right_margin_,
                                overflow_, std::next(line_), end_);
  }

  bool equal(alignment_line const &other) const {
    return line_ == other.line_ && overflow_ == other.overflow_;
  }

  Alignment<Block> const *alignment_ = nullptr;
  std::size_t left_margin_ = 0, right_margin_ = 0;
  std::ptrdiff_t overflow_ = 0;
  block_iterator line_{}, end_{};

public:
  alignment_line() = default;
};

template <typename Block>
inline std::ostream &operator<<(std::ostream &stream,
                                alignment_line<Block> const &line) {
  auto fill = stream.fill();
  stream << std::setfill(' ');
  if (!line.overflow_ && line.line_ != line.end_) {
    stream << std::setw(line.left_margin_) << u8"" << *line.line_
           << std::setw(line.right_margin_) << u8"";
  } else {
    stream << std::setw(line.alignment_->size().x) << u8"";
  }

  return stream << std::setfill(fill);
}
} /* namespace detail */

template <typename Block>
class Alignment {
public:
  using value_type = detail::alignment_line<Block>;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<Alignment<Block>, value_type>;
  using iterator = const_iterator;
  using difference_type = typename const_iterator::difference_type;
  using size_type = Size;

  explicit Alignment(Block block) : block_(std::move(block)) {}

  explicit Alignment(Size sz, Block block)
      : size_(sz), block_(std::move(block)) {}

  explicit Alignment(Align halign, Size sz, Block block)
      : halign_(halign), size_(sz), block_(std::move(block)) {}

  explicit Alignment(VAlign valign, Size sz, Block block)
      : valign_(valign), size_(sz), block_(std::move(block)) {}

  explicit Alignment(Align halign, VAlign valign, Size sz, Block block)
      : halign_(halign), valign_(valign), size_(sz), block_(std::move(block)) {}

  Size size() const {
    auto block_sz = detail::block_traits<Block>::size(block_);
    return {utils::max(block_sz.x, size_.x), utils::max(block_sz.y, size_.y)};
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const {
    auto sz = size();
    auto block_sz = detail::block_traits<Block>::size(block_);

    auto hmargin = sz.x - block_sz.x, vmargin = sz.y - block_sz.y;
    auto top = (valign_ == VAlign::Middle)
                   ? vmargin / 2
                   : (valign_ == VAlign::Bottom) ? vmargin : 0;
    auto left = (halign_ == Align::Center)
                    ? hmargin / 2
                    : (halign_ == Align::Right) ? hmargin : 0;

    return {{this, std::size_t(left), std::size_t(hmargin - left),
             -std::ptrdiff_t(top), detail::block_traits<Block>::begin(block_),
             detail::block_traits<Block>::end(block_)}};
  }

  const_iterator cend() const {
    auto sz = size();
    auto block_sz = detail::block_traits<Block>::size(block_);

    auto hmargin = sz.x - block_sz.x, vmargin = sz.y - block_sz.y;
    auto top = (valign_ == VAlign::Middle)
                   ? vmargin / 2
                   : (valign_ == VAlign::Bottom) ? vmargin : 0;
    auto left = (halign_ == Align::Center)
                    ? hmargin / 2
                    : (halign_ == Align::Right) ? hmargin : 0;

    return {{this, std::size_t(left), std::size_t(hmargin - left),
             std::ptrdiff_t(vmargin - top),
             detail::block_traits<Block>::end(block_),
             detail::block_traits<Block>::end(block_)}};
  }

private:
  friend std::ostream &detail::operator<<<Block>(std::ostream &,
                                                 value_type const &);

  Align halign_ = Align::Center;
  VAlign valign_ = VAlign::Middle;
  Size size_ = {0, 0};
  Block block_;
};

template <typename Block>
inline std::ostream &operator<<(std::ostream &stream,
                                Alignment<Block> const &alignment) {
  for (auto const &line : alignment)
    stream << line << '\n';

  return stream;
}

template <typename Block>
inline Alignment<std::decay_t<Block>> alignment(Block &&block) {
  return Alignment<std::decay_t<Block>>(std::forward<Block>(block));
}

template <typename Block>
inline Alignment<std::decay_t<Block>> alignment(Size sz, Block &&block) {
  return Alignment<std::decay_t<Block>>(sz, std::forward<Block>(block));
}

template <typename Block>
inline Alignment<std::decay_t<Block>> alignment(Align halign, Size sz,
                                                Block &&block) {
  return Alignment<std::decay_t<Block>>(halign, sz, std::forward<Block>(block));
}

template <typename Block>
inline Alignment<std::decay_t<Block>> alignment(VAlign valign, Size sz,
                                                Block &&block) {
  return Alignment<std::decay_t<Block>>(valign, sz, std::forward<Block>(block));
}

template <typename Block>
inline Alignment<std::decay_t<Block>> alignment(Align halign, VAlign valign,
                                                Size sz, Block &&block) {
  return Alignment<std::decay_t<Block>>(halign, valign, sz,
                                        std::forward<Block>(block));
}

template <typename Block>
class Margin;

namespace detail {
template <typename Block>
class margin_line;

template <typename Block>
std::ostream &operator<<(std::ostream &, margin_line<Block> const &);

template <typename Block>
class margin_line {
  using block_iterator = typename detail::block_traits<Block>::iterator;

  friend class detail::block_iterator<Margin<Block>, margin_line>;
  friend class Margin<Block>;

  friend std::ostream &operator<<<Block>(std::ostream &, margin_line const &);

  margin_line(Margin<Block> const *margin, std::ptrdiff_t overflow,
              block_iterator line, block_iterator end)
      : margin_(margin), overflow_(overflow), line_(std::move(line)),
        end_(std::move(end)) {}

  margin_line next() const {
    return (overflow_ || line_ == end_)
               ? margin_line(margin_, overflow_ + 1, line_, end_)
               : margin_line(margin_, overflow_, std::next(line_), end_);
  }

  bool equal(margin_line const &other) const {
    return line_ == other.line_ && overflow_ == other.overflow_;
  }

  Margin<Block> const *margin_ = nullptr;
  std::ptrdiff_t overflow_ = 0;
  block_iterator line_{}, end_{};

public:
  margin_line() = default;
};

template <typename Block>
inline std::ostream &operator<<(std::ostream &stream,
                                margin_line<Block> const &line) {
  auto fill = stream.fill();
  stream << std::setfill(' ');
  if (!line.overflow_ && line.line_ != line.end_) {
    stream << std::setw(line.margin_->left_) << u8"" << *line.line_
           << std::setw(line.margin_->right_) << u8"";
  } else {
    stream << std::setw(line.margin_->size().x) << u8"";
  }

  return stream << std::setfill(fill);
}
} /* namespace detail */

template <typename Block>
class Margin {
public:
  using value_type = detail::margin_line<Block>;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<Margin<Block>, value_type>;
  using iterator = const_iterator;
  using difference_type = typename const_iterator::difference_type;
  using size_type = Size;

  explicit Margin(Block block) : block_(std::move(block)) {}

  explicit Margin(std::size_t margin, Block block)
      : top_(margin), right_(margin), bottom_(margin), left_(margin),
        block_(std::move(block)) {}

  explicit Margin(std::size_t v, std::size_t h, Block block)
      : top_(v), right_(h), bottom_(v), left_(h), block_(std::move(block)) {}

  explicit Margin(std::size_t top, std::size_t right, std::size_t bottom,
                  std::size_t left, Block block)
      : top_(top), right_(right), bottom_(bottom), left_(left),
        block_(std::move(block)) {}

  Size size() const {
    return detail::block_traits<Block>::size(block_) +
           Size(left_ + right_, top_ + bottom_);
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const {
    return {{this, -std::ptrdiff_t(top_),
             detail::block_traits<Block>::begin(block_),
             detail::block_traits<Block>::end(block_)}};
  }

  const_iterator cend() const {
    return {{this, std::ptrdiff_t(bottom_),
             detail::block_traits<Block>::end(block_),
             detail::block_traits<Block>::end(block_)}};
  }

private:
  friend std::ostream &detail::operator<<<Block>(std::ostream &,
                                                 value_type const &);

  std::size_t top_ = 1, right_ = 2, bottom_ = 1, left_ = 2;
  Block block_;
};

template <typename Block>
inline std::ostream &operator<<(std::ostream &stream,
                                Margin<Block> const &margin) {
  for (auto const &line : margin)
    stream << line << '\n';

  return stream;
}

template <typename Block>
inline Margin<std::decay_t<Block>> margin(Block &&block) {
  return Margin<std::decay_t<Block>>(std::forward<Block>(block));
}

template <typename Block>
inline Margin<std::decay_t<Block>> margin(std::size_t margin, Block &&block) {
  return Margin<std::decay_t<Block>>(margin, std::forward<Block>(block));
}

template <typename Block>
inline Margin<std::decay_t<Block>> margin(std::size_t v, std::size_t h,
                                          Block &&block) {
  return Margin<std::decay_t<Block>>(v, h, std::forward<Block>(block));
}

template <typename Block>
inline Margin<std::decay_t<Block>> margin(std::size_t top, std::size_t right,
                                          std::size_t bottom, std::size_t left,
                                          Block &&block) {
  return Margin<std::decay_t<Block>>(top, right, bottom, left,
                                     std::forward<Block>(block));
}

template <typename Block>
class Frame;

namespace detail {
template <typename Block>
class frame_line;

template <typename Block>
std::ostream &operator<<(std::ostream &, frame_line<Block> const &);

template <typename Block>
class frame_line {
  using block_iterator = typename detail::block_traits<Block>::iterator;

  friend class detail::block_iterator<Frame<Block>, frame_line>;
  friend class Frame<Block>;

  friend std::ostream &operator<<<Block>(std::ostream &, frame_line const &);

  frame_line(Frame<Block> const *frame, std::ptrdiff_t overflow,
             block_iterator line, block_iterator end)
      : frame_(frame), overflow_(overflow), line_(std::move(line)),
        end_(std::move(end)) {}

  frame_line next() const {
    return (overflow_ || line_ == end_)
               ? frame_line(frame_, overflow_ + 1, line_, end_)
               : frame_line(frame_, overflow_, std::next(line_), end_);
  }

  bool equal(frame_line const &other) const {
    return line_ == other.line_ && overflow_ == other.overflow_;
  }

  Frame<Block> const *frame_ = nullptr;
  std::ptrdiff_t overflow_ = 0;
  block_iterator line_{}, end_{};

public:
  frame_line() = default;
};
} /* namespace detail */

template <typename Block>
class Frame {
public:
  using value_type = detail::frame_line<Block>;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<Frame<Block>, value_type>;
  using iterator = const_iterator;
  using difference_type = typename const_iterator::difference_type;
  using size_type = Size;

  explicit Frame(Block block, TerminalInfo term = TerminalInfo())
      : block_(std::move(block)), term_(term) {}

  explicit Frame(Border border, Block block, TerminalInfo term = TerminalInfo())
      : border_(border), block_(std::move(block)), term_(term) {}

  explicit Frame(std::string_view label, Block block,
                 TerminalInfo term = TerminalInfo())
      : label_(label), block_(std::move(block)), term_(term) {}

  explicit Frame(std::string_view label, Align align, Block block,
                 TerminalInfo term = TerminalInfo())
      : label_(label), align_(align), block_(std::move(block)), term_(term) {}

  explicit Frame(std::string_view label, Border border, Block block,
                 TerminalInfo term = TerminalInfo())
      : label_(label), border_(border), block_(std::move(block)), term_(term) {}

  explicit Frame(std::string_view label, Align align, Border border,
                 Block block, TerminalInfo term = TerminalInfo())
      : label_(label), align_(align), border_(border), block_(std::move(block)),
        term_(term) {}

  Size size() const {
    return detail::block_traits<Block>::size(block_) + Size(2, 2);
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const {
    return {{this, -1, detail::block_traits<Block>::begin(block_),
             detail::block_traits<Block>::end(block_)}};
  }

  const_iterator cend() const {
    return {{this, 1, detail::block_traits<Block>::end(block_),
             detail::block_traits<Block>::end(block_)}};
  }

private:
  template <typename>
  friend class detail::frame_line;
  friend std::ostream &detail::operator<<<Block>(std::ostream &,
                                                 value_type const &);

  std::string_view label_;
  Align align_ = Align::Left;
  Border border_{BorderStyle::Solid};
  Block block_;
  TerminalInfo term_;
};

template <typename Block>
inline std::ostream &operator<<(std::ostream &stream,
                                Frame<Block> const &frame) {
  for (auto const &line : frame)
    stream << line << '\n';

  return stream;
}

namespace detail {
template <typename Block>
std::ostream &operator<<(std::ostream &stream, frame_line<Block> const &line) {
  auto size = detail::block_traits<Block>::size(line.frame_->block_);
  auto const border = line.frame_->border_;

  if (line.overflow_ < 0) {
    auto lbl =
        label(line.frame_->label_, line.frame_->align_, size.x, border.top);
    auto lbl_line = *lbl.cbegin();

    return stream << line.frame_->term_.reset() << border.top_left << lbl_line
                  << border.top_right;
  } else if (line.line_ == line.end_) {
    stream << line.frame_->term_.reset() << border.bottom_left;

    for (auto i = 0; i < size.x; ++i)
      stream << border.bottom;

    return stream << border.bottom_right;
  }

  return stream << line.frame_->term_.reset() << line.frame_->border_.left
                << *line.line_ << line.frame_->term_.reset()
                << line.frame_->border_.right;
}
} /* namespace detail */

template <typename Block>
inline Frame<std::decay_t<Block>> frame(Block &&block,
                                        TerminalInfo term = TerminalInfo()) {
  return Frame<std::decay_t<Block>>(std::forward<Block>(block), term);
}

template <typename Block>
inline Frame<std::decay_t<Block>> frame(Border border, Block &&block,
                                        TerminalInfo term = TerminalInfo()) {
  return Frame<std::decay_t<Block>>(border, std::forward<Block>(block), term);
}

template <typename Block>
inline Frame<std::decay_t<Block>> frame(std::string_view label, Block &&block,
                                        TerminalInfo term = TerminalInfo()) {
  return Frame<std::decay_t<Block>>(label, std::forward<Block>(block), term);
}

template <typename Block>
inline Frame<std::decay_t<Block>> frame(std::string_view label, Align align,
                                        Block &&block,
                                        TerminalInfo term = TerminalInfo()) {
  return Frame<std::decay_t<Block>>(label, align, std::forward<Block>(block),
                                    term);
}

template <typename Block>
inline Frame<std::decay_t<Block>> frame(std::string_view label, Border border,
                                        Block &&block,
                                        TerminalInfo term = TerminalInfo()) {
  return Frame<std::decay_t<Block>>(label, border, std::forward<Block>(block),
                                    term);
}

template <typename Block>
inline Frame<std::decay_t<Block>> frame(std::string_view label, Align align,
                                        Border border, Block &&block,
                                        TerminalInfo term = TerminalInfo()) {
  return Frame<std::decay_t<Block>>(label, align, border,
                                    std::forward<Block>(block), term);
}

template <typename... Blocks>
class VBox;

namespace detail {
struct plus_identity {};

template <typename Arg>
inline constexpr decltype(auto) operator+(plus_identity, Arg &&arg) {
  return std::forward<Arg>(arg);
}

template <typename Arg>
inline constexpr decltype(auto) operator+(Arg &&arg, plus_identity) {
  return std::forward<Arg>(arg);
}

inline constexpr plus_identity plus_fold() { return {}; }

template <typename Arg, typename... Args>
inline constexpr auto plus_fold(Arg &&first, Args &&... rest) {
  return std::forward<Arg>(first) + plus_fold(std::forward<Args>(rest)...);
}

inline constexpr std::size_t find_true(std::size_t index) { return index; }

template <typename... Args>
inline constexpr std::size_t find_true(std::size_t index, bool first,
                                       Args &&... rest) {
  return first ? index : find_true(index + 1, std::forward<Args>(rest)...);
}

template <typename... Args>
inline constexpr std::size_t find_true(bool first, Args &&... rest) {
  return first ? 0 : find_true(std::size_t(1), std::forward<Args>(rest)...);
}

template <typename... Blocks>
class vbox_line;

template <typename... Blocks>
std::ostream &operator<<(std::ostream &, vbox_line<Blocks...> const &);

template <typename... Blocks>
class vbox_line {
  using block_iterators =
      std::tuple<typename detail::block_traits<Blocks>::iterator...>;

  friend class detail::block_iterator<VBox<Blocks...>, vbox_line>;
  friend class VBox<Blocks...>;

  friend std::ostream &operator<<<Blocks...>(std::ostream &, vbox_line const &);

  vbox_line(VBox<Blocks...> const *vbox, std::size_t margin,
            block_iterators lines, block_iterators ends)
      : vbox_(vbox), margin_(margin), lines_(std::move(lines)),
        ends_(std::move(ends)) {}

  vbox_line next() const {
    return margin_ ? vbox_line(vbox_, margin_ - 1, lines_, ends_)
                   : next_impl(std::make_index_sequence<sizeof...(Blocks)>());
  }

  bool equal(vbox_line const &other) const {
    return lines_ == other.lines_ && margin_ == other.margin_;
  }

  template <std::size_t... N>
  vbox_line next_impl(std::index_sequence<N...> indices) const {
    auto current = current_index(indices);
    block_iterators nxt(((N != current) ? std::get<N>(lines_)
                                        : std::next(std::get<N>(lines_)))...);
    return {vbox_,
            (find_true((std::get<N>(nxt) != std::get<N>(ends_))...) != current)
                ? vbox_->margin_
                : 0,
            nxt, ends_};
  }

  template <std::size_t... N>
  std::size_t current_index(std::index_sequence<N...>) const {
    return find_true((std::get<N>(lines_) != std::get<N>(ends_))...);
  }

  VBox<Blocks...> const *vbox_;
  std::size_t margin_ = 0;
  block_iterators lines_, ends_;

public:
  vbox_line() = default;
};

inline std::ostream &output_vbox_line(std::ostream &stream, std::size_t) {
  return stream;
}

template <typename Arg, typename... Args>
inline std::ostream &output_vbox_line(std::ostream &stream, std::size_t width,
                                      Arg &&first, Args &&... rest) {
  return (std::get<0>(std::forward<Arg>(first)) !=
          std::get<1>(std::forward<Arg>(first)))
             ? (stream << *std::get<0>(std::forward<Arg>(first))
                       << std::setw(
                              width -
                              detail::block_traits<std::decay_t<
                                  std::tuple_element_t<2, std::decay_t<Arg>>>>::
                                  size(std::get<2>(std::forward<Arg>(first)))
                                      .x)
                       << u8"")
             : output_vbox_line(stream, width, std::forward<Args>(rest)...);
}

template <typename Iterators, typename Blocks, std::size_t... N>
inline std::ostream &
output_vbox_line(std::ostream &stream, std::size_t width,
                 Iterators const &lines, Iterators const &ends,
                 Blocks const &blocks, std::index_sequence<N...>) {
  return output_vbox_line(stream, width,
                          std::forward_as_tuple(std::get<N>(lines),
                                                std::get<N>(ends),
                                                std::get<N>(blocks))...);
}

template <typename... Blocks>
inline std::ostream &operator<<(std::ostream &stream,
                                vbox_line<Blocks...> const &line) {
  auto fill = stream.fill();
  auto width = line.vbox_->size().x;

  stream << std::setfill(' ');

  if (!line.margin_)
    output_vbox_line(stream, width, line.lines_, line.ends_,
                     line.vbox_->blocks_,
                     std::make_index_sequence<sizeof...(Blocks)>());
  else
    stream << std::setw(width) << u8"";

  return stream << std::setfill(fill);
}
} /* namespace detail */

template <typename... Blocks>
class VBox {
public:
  using value_type = detail::vbox_line<Blocks...>;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<VBox<Blocks...>, value_type>;
  using iterator = const_iterator;
  using difference_type = typename const_iterator::difference_type;
  using size_type = Size;

  explicit VBox(Blocks... blocks) : blocks_(std::move(blocks)...) {}

  explicit VBox(std::size_t margin, Blocks... blocks)
      : margin_(margin), blocks_(std::move(blocks)...) {}

  Size size() const {
    return size_impl(std::make_index_sequence<sizeof...(Blocks)>()) +
           Size(0, margin_ * (sizeof...(Blocks) - 1));
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const {
    return {{this, 0, begins(std::make_index_sequence<sizeof...(Blocks)>()),
             ends(std::make_index_sequence<sizeof...(Blocks)>())}};
  }

  const_iterator cend() const {
    auto e = ends(std::make_index_sequence<sizeof...(Blocks)>());
    return {{this, margin_, e, e}};
  }

private:
  friend value_type;

  friend std::ostream &detail::operator<<<Blocks...>(std::ostream &,
                                                     value_type const &);

  template <std::size_t... N>
  Size size_impl(std::index_sequence<N...>) const {
    return {
        std::max(
            {detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::
                 size(std::get<N>(blocks_))
                     .x...}),
        detail::plus_fold(
            detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::
                size(std::get<N>(blocks_))
                    .y...)};
  }

  template <std::size_t... N>
  typename value_type::block_iterators begins(std::index_sequence<N...>) const {
    return typename value_type::block_iterators{
        detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::begin(
            std::get<N>(blocks_))...};
  }

  template <std::size_t... N>
  typename value_type::block_iterators ends(std::index_sequence<N...>) const {
    return typename value_type::block_iterators{
        detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::end(
            std::get<N>(blocks_))...};
  }

  std::size_t margin_ = 1;
  std::tuple<Blocks...> blocks_;
};

template <typename Block, typename... Blocks>
inline std::enable_if_t<
    !std::is_convertible<std::decay_t<Block>, std::size_t>::value,
    VBox<std::decay_t<Block>, std::decay_t<Blocks>...>>
vbox(Block &&block, Blocks &&... blocks) {
  return VBox<std::decay_t<Block>, std::decay_t<Blocks>...>(
      std::forward<Block>(block), std::forward<Blocks>(blocks)...);
}

template <typename Margin, typename... Blocks>
inline std::enable_if_t<
    std::is_convertible<std::decay_t<Margin>, std::size_t>::value,
    VBox<std::decay_t<Blocks>...>>
vbox(Margin &&margin, Blocks &&... blocks) {
  return VBox<std::decay_t<Blocks>...>(std::forward<Margin>(margin),
                                       std::forward<Blocks>(blocks)...);
}

template <typename... Blocks>
class HBox;

namespace detail {
template <typename... Blocks>
class hbox_line;

template <typename... Blocks>
std::ostream &operator<<(std::ostream &, hbox_line<Blocks...> const &);

template <typename... Blocks>
class hbox_line {
  using block_iterators =
      std::tuple<typename detail::block_traits<Blocks>::iterator...>;

  friend class detail::block_iterator<HBox<Blocks...>, hbox_line>;
  friend class HBox<Blocks...>;

  friend std::ostream &operator<<<Blocks...>(std::ostream &, hbox_line const &);

  hbox_line(HBox<Blocks...> const *hbox, std::size_t margin,
            block_iterators lines, block_iterators ends)
      : hbox_(hbox), margin_(margin), lines_(std::move(lines)),
        ends_(std::move(ends)) {}

  hbox_line next() const {
    return next_impl(std::make_index_sequence<sizeof...(Blocks)>());
  }

  bool equal(hbox_line const &other) const { return lines_ == other.lines_; }

  template <std::size_t... N>
  hbox_line next_impl(std::index_sequence<N...>) const {
    return {hbox_, margin_,
            block_iterators{((std::get<N>(lines_) != std::get<N>(ends_))
                                 ? std::next(std::get<N>(lines_))
                                 : std::get<N>(lines_))...},
            ends_};
  }

  HBox<Blocks...> const *hbox_;
  std::size_t margin_ = 0;
  block_iterators lines_, ends_;

public:
  hbox_line() = default;
};

inline std::ostream &output_hbox_line(std::ostream &stream, std::size_t) {
  return stream;
}

template <typename Arg, typename... Args>
inline std::ostream &output_hbox_line(std::ostream &stream, std::size_t margin,
                                      Arg &&first, Args &&... rest) {
  return output_hbox_line(
      ((std::get<0>(std::forward<Arg>(first)) !=
        std::get<1>(std::forward<Arg>(first)))
           ? stream << std::setw(margin) << u8""
                    << *std::get<0>(std::forward<Arg>(first))
           : stream << std::setw(
                           margin +
                           detail::block_traits<std::decay_t<
                               std::tuple_element_t<2, std::decay_t<Arg>>>>::
                               size(std::get<2>(std::forward<Arg>(first)))
                                   .x)
                    << u8""),
      margin, std::forward<Args>(rest)...);
}

template <typename Iterators, typename Blocks, std::size_t... N>
inline std::ostream &
output_hbox_line(std::ostream &stream, std::size_t margin,
                 Iterators const &lines, Iterators const &ends,
                 Blocks const &blocks, std::index_sequence<N...>) {
  return output_hbox_line(
      ((std::get<0>(lines) != std::get<0>(ends))
           ? stream << *std::get<0>(lines)
           : stream << std::setw(
                           detail::block_traits<
                               std::decay_t<std::tuple_element_t<0, Blocks>>>::
                               size(std::get<0>(blocks))
                                   .x)
                    << u8""),
      margin,
      std::forward_as_tuple(std::get<N + 1>(lines), std::get<N + 1>(ends),
                            std::get<N + 1>(blocks))...);
}

template <typename... Blocks>
inline std::ostream &operator<<(std::ostream &stream,
                                hbox_line<Blocks...> const &line) {
  auto fill = stream.fill();
  return output_hbox_line(stream << std::setfill(' '), line.margin_,
                          line.lines_, line.ends_, line.hbox_->blocks_,
                          std::make_index_sequence<sizeof...(Blocks) - 1>())
         << std::setfill(fill);
}
} /* namespace detail */

template <typename... Blocks>
class HBox {
public:
  using value_type = detail::hbox_line<Blocks...>;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<HBox<Blocks...>, value_type>;
  using iterator = const_iterator;
  using difference_type = typename const_iterator::difference_type;
  using size_type = Size;

  explicit HBox(Blocks... blocks) : blocks_(std::move(blocks)...) {}

  explicit HBox(std::size_t margin, Blocks... blocks)
      : margin_(margin), blocks_(std::move(blocks)...) {}

  Size size() const {
    return size_impl(std::make_index_sequence<sizeof...(Blocks)>()) +
           Size(margin_ * (sizeof...(Blocks) - 1), 0);
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const {
    return {{this, margin_,
             begins(std::make_index_sequence<sizeof...(Blocks)>()),
             ends(std::make_index_sequence<sizeof...(Blocks)>())}};
  }

  const_iterator cend() const {
    auto e = ends(std::make_index_sequence<sizeof...(Blocks)>());
    return {{this, margin_, e, e}};
  }

private:
  friend std::ostream &detail::operator<<<Blocks...>(std::ostream &,
                                                     value_type const &);

  template <std::size_t... N>
  Size size_impl(std::index_sequence<N...>) const {
    return {
        detail::plus_fold(
            detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::
                size(std::get<N>(blocks_))
                    .x...),
        std::max(
            {detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::
                 size(std::get<N>(blocks_))
                     .y...}),
    };
  }

  template <std::size_t... N>
  typename value_type::block_iterators begins(std::index_sequence<N...>) const {
    return typename value_type::block_iterators{
        detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::begin(
            std::get<N>(blocks_))...};
  }

  template <std::size_t... N>
  typename value_type::block_iterators ends(std::index_sequence<N...>) const {
    return typename value_type::block_iterators{
        detail::block_traits<std::tuple_element_t<N, decltype(blocks_)>>::end(
            std::get<N>(blocks_))...};
  }

  std::size_t margin_ = 2;
  std::tuple<Blocks...> blocks_;
};

template <typename Block, typename... Blocks>
inline std::enable_if_t<
    !std::is_convertible<std::decay_t<Block>, std::size_t>::value,
    HBox<std::decay_t<Block>, std::decay_t<Blocks>...>>
hbox(Block &&block, Blocks &&... blocks) {
  return HBox<std::decay_t<Block>, std::decay_t<Blocks>...>(
      std::forward<Block>(block), std::forward<Blocks>(blocks)...);
}

template <typename Margin, typename... Blocks>
inline std::enable_if_t<
    std::is_convertible<std::decay_t<Margin>, std::size_t>::value,
    HBox<std::decay_t<Blocks>...>>
hbox(Margin &&margin, Blocks &&... blocks) {
  return HBox<std::decay_t<Blocks>...>(std::forward<Margin>(margin),
                                       std::forward<Blocks>(blocks)...);
}

} /* namespace plot */

// include/braille.hpp

namespace plot {

class BrailleCanvas;

namespace detail {
namespace braille {
// The dimensions of a Braille cell are 2x4
constexpr std::uint8_t cell_cols = 2;
constexpr std::uint8_t cell_rows = 4;
// Unicode braille patterns: 0x28xx
// See https://en.wikipedia.org/wiki/Braille_Patterns
static constexpr std::uint8_t pixel_codes[cell_cols][cell_rows] = {
    {0x01, 0x02, 0x04, 0x40}, {0x08, 0x10, 0x20, 0x80}};

inline constexpr std::uint8_t bitcount(std::uint8_t n) {
  return (n & 1) + bool(n & 2) + bool(n & 4) + bool(n & 8) + bool(n & 16) +
         bool(n & 32) + bool(n & 64) + bool(n & 128);
}

struct block_t {
  constexpr block_t() = default;

  constexpr block_t(Color c, bool px00, bool px01, bool px02, bool px03,
                    bool px10, bool px11, bool px12, bool px13)
      : color(c), pixels(pixel_codes[0][0] * px00 | pixel_codes[0][1] * px01 |
                         pixel_codes[0][2] * px02 | pixel_codes[0][3] * px03 |
                         pixel_codes[1][0] * px10 | pixel_codes[1][1] * px11 |
                         pixel_codes[1][2] * px12 | pixel_codes[1][3] * px13) {}

  constexpr block_t(Color c, std::uint8_t px = 0) : color(c), pixels(px) {}

  block_t &clear() {
    pixels = 0;
    return *this;
  }

  block_t &clear(std::size_t x, std::size_t y) {
    pixels &= ~pixel_codes[x % cell_cols][y % cell_rows];
    return *this;
  }

  block_t &set(std::size_t x, std::size_t y) {
    pixels |= pixel_codes[x % cell_cols][y % cell_rows];
    return *this;
  }

  block_t over(block_t const &other) const {
    auto old = bitcount(other.pixels & ~pixels);
    auto new_ = bitcount(pixels & ~other.pixels);

    std::uint8_t over_pixels = other.pixels & pixels;
    auto over_ = bitcount(over_pixels);

    float total = old + new_ + over_;

    auto old_color = (other.color.a != 0.0f) ? other.color : color;
    auto new_color = (color.a != 0.0f) ? color : other.color;
    auto over_color = new_color.over(old_color);

    auto mixed_color = (old / total) * old_color + (new_ / total) * new_color +
                       (over_ / total) * over_color;

    return {mixed_color, std::uint8_t(pixels | other.pixels)};
  }

  block_t paint(block_t const &dst, TerminalOp op) const {
    if (pixels) {
      switch (op) {
      case TerminalOp::Over:
        return over(dst);
      case TerminalOp::ClipDst:
        return *this;
      case TerminalOp::ClipSrc:
        if (!dst.pixels)
          return *this;
      }
    }

    return dst;
  }

  constexpr block_t operator~() const { return {color, std::uint8_t(~pixels)}; }

  constexpr block_t operator|(block_t const &other) const {
    return {color, std::uint8_t(pixels | other.pixels)};
  }

  block_t &operator|=(block_t const &other) {
    return (*this) = (*this) | other;
  }

  constexpr block_t operator&(block_t const &other) const {
    return {color, std::uint8_t(pixels & other.pixels)};
  }

  block_t &operator&=(block_t const &other) {
    return (*this) = (*this) & other;
  }

  Color color{0, 0, 0, 0};
  std::uint8_t pixels = 0;
};

class image_t : public std::vector<block_t> {
  using base = std::vector<block_t>;

public:
  using base::base;

  image_t(Size sz) : base(sz.y * sz.x) {}

  void clear() { assign(size(), block_t()); }

  void resize(Size from, Size to) {
    if (std::size_t(to.y * to.x) > size())
      resize(to.y * to.x, block_t());

    auto first = begin();

    if (to.x < from.x) {
      for (Coord line = 1, end_ = std::min(to.y, from.y); line < end_; ++line)
        std::copy(first + line * from.x, first + line * from.x + to.x,
                  first + line * to.x);
    } else if (to.x > from.x) {
      for (Coord line = std::min(to.y, from.y) - 1; line > 0; --line) {
        std::copy_backward(first + line * from.x,
                           first + line * from.x + from.x,
                           first + line * to.x + from.x);
        std::fill(first + line * from.x, first + line * to.x, block_t());
      }
    }

    if (std::size_t(to.y * to.x) < size())
      resize(to.y * to.x);
  }

  // XXX: undefined behavior if this and other do not have the same layout
  void paint(image_t const &other, TerminalOp op) {
    auto dst = begin();
    for (auto src = other.begin(), src_end = other.end(); src != src_end;
         ++src, ++dst)
      *dst = src->paint(*dst, op);
  }

private:
  using base::resize;
};

class line_t;

template <typename = void>
std::ostream &operator<<(std::ostream &stream, line_t const &line);

class line_t {
  friend class detail::block_iterator<plot::BrailleCanvas, line_t>;
  friend class plot::BrailleCanvas;

  template <typename>
  friend std::ostream &operator<<(std::ostream &, line_t const &);

  line_t(BrailleCanvas const *canvas, image_t::const_iterator it)
      : canvas_(canvas), it_(it) {}

  line_t next() const;

  bool equal(line_t const &other) const { return it_ == other.it_; }

  BrailleCanvas const *canvas_ = nullptr;
  image_t::const_iterator it_{};

public:
  line_t() = default;
};
} /* namespace braille */
} /* namespace detail */

class BrailleCanvas {
public:
  constexpr static uint8_t cell_cols = detail::braille::cell_cols;
  constexpr static uint8_t cell_rows = detail::braille::cell_rows;
  using value_type = detail::braille::line_t;
  using reference = value_type const &;
  using const_reference = value_type const &;
  using const_iterator = detail::block_iterator<BrailleCanvas, value_type>;
  using iterator = const_iterator;
  using difference_type = const_iterator::difference_type;

  using coord_type = Coord;
  using point_type = Point;
  using size_type = Size;
  using rect_type = Rect;

  BrailleCanvas() = default;

  BrailleCanvas(Size char_sz, TerminalInfo term = TerminalInfo())
      : lines_(char_sz.y), cols_(char_sz.x), blocks_(char_sz),
        background_(term.background_color), term_(term) {
    available_layers_.emplace_front(char_sz);
  }

  BrailleCanvas(Color background, Size char_sz,
                TerminalInfo term = TerminalInfo())
      : lines_(char_sz.y), cols_(char_sz.x), blocks_(char_sz),
        background_(background), term_(term) {
    available_layers_.emplace_front(char_sz);
  }

  Size char_size() const { return {Coord(cols_), Coord(lines_)}; }

  Size size() const {
    return {Coord(cell_cols * cols_), Coord(cell_rows * lines_)};
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  const_iterator cbegin() const { return {{this, blocks_.cbegin()}}; }

  const_iterator cend() const { return {{this, blocks_.cend()}}; }

  BrailleCanvas &push() {
    if (available_layers_.empty())
      available_layers_.emplace_front(char_size());

    stack_.splice_after(stack_.before_begin(), available_layers_,
                        available_layers_.before_begin());
    blocks_.swap(stack_.front());
    blocks_.clear();
    return *this;
  }

  BrailleCanvas &pop(TerminalOp op = TerminalOp::Over) {
    if (!stack_.empty()) {
      stack_.front().paint(blocks_, op);
      blocks_.swap(stack_.front());
      available_layers_.splice_after(available_layers_.before_begin(), stack_,
                                     stack_.before_begin());
    }
    return *this;
  }

  BrailleCanvas &resize(Size sz) {
    if (sz != char_size()) {
      blocks_.resize(char_size(), sz);

      for (auto &layer : stack_)
        layer.resize(char_size(), sz);

      if (!available_layers_.empty()) {
        available_layers_.clear();
        available_layers_.emplace_front(sz);
      }

      lines_ = sz.y;
      cols_ = sz.x;
    }
    return *this;
  }

  BrailleCanvas &clear() {
    blocks_.clear();
    return *this;
  }

  BrailleCanvas &clear(Color background) {
    this->background_ = background;
    return clear();
  }

  BrailleCanvas &clear(Rect rct) {
    rct = rct.sorted();
    Rect block_rect{
        {rct.p1.x / cell_cols, rct.p1.y / cell_rows},
        {utils::max(1l, rct.p2.x / cell_cols + (rct.p2.x % cell_cols)),
         utils::max(1l, rct.p2.y / cell_rows + (rct.p2.y % cell_rows != 0))}};

    rct.p2 += Point(1, 1);

    for (auto ln = block_rect.p1.y; ln < block_rect.p2.y; ++ln) {
      auto ybase = cell_rows * ln;
      for (auto col = block_rect.p1.x; col < block_rect.p2.x; ++col) {
        auto xbase = cell_cols * col;
        detail::braille::block_t src(
            {0, 0, 0, 0}, rct.contains({xbase, ybase}),
            rct.contains({xbase, ybase + 1}), rct.contains({xbase, ybase + 2}),
            rct.contains({xbase, ybase + 3}), rct.contains({xbase + 1, ybase}),
            rct.contains({xbase + 1, ybase + 1}),
            rct.contains({xbase + 1, ybase + 2}),
            rct.contains({xbase + 1, ybase + 3}));
        block(ln, col) &= ~src;
      }
    }

    return *this;
  }

  template <typename Fn>
  BrailleCanvas &stroke(Color const &color, Rect rct, Fn &&fn,
                        TerminalOp op = TerminalOp::Over);

  template <typename Fn>
  BrailleCanvas &fill(Color const &color, Rect rct, Fn &&fn,
                      TerminalOp op = TerminalOp::Over);

  BrailleCanvas &dot(Color const &color, Point p,
                     TerminalOp op = TerminalOp::Over) {
    if (Rect({}, size()).contains(p)) {
      paint(
          p.y / cell_rows, p.x / cell_cols,
          detail::braille::block_t(color).set(p.x % cell_cols, p.y % cell_rows),
          op);
    }
    return *this;
  }

  BrailleCanvas &line(Color const &color, Point from, Point to,
                      TerminalOp op = TerminalOp::Over) {
    auto sorted = Rect(from, to).sorted_x();
    auto dx = (sorted.p2.x - sorted.p1.x) + 1, dy = sorted.p2.y - sorted.p1.y;

    dy += (dy >= 0) - (dy < 0);

    auto gcd = utils::gcd(dx, dy);
    dx /= gcd;
    dy /= gcd;

    return stroke(
        color, sorted,
        [dx, dy, x0 = sorted.p1.x, y0 = sorted.p1.y](Coord x) {
          auto base = (x - x0) * dy / dx + y0,
               end_ = (1 + x - x0) * dy / dx + y0;
          return (base != end_) ? std::make_pair(base, end_)
                                : std::make_pair(base, base + 1);
        },
        op);
  }

  template <typename Iterator>
  BrailleCanvas &path(Color const &color, Iterator first, Iterator last,
                      TerminalOp op = TerminalOp::Over) {
    push();
    auto start = *first;
    while (++first != last) {
      auto end_ = *first;
      line(color, start, end_, TerminalOp::Over);
      start = end_;
    }
    return pop(op);
  }

  BrailleCanvas &path(Color const &color,
                      std::initializer_list<Point> const &points,
                      TerminalOp op = TerminalOp::Over) {
    return path(color, points.begin(), points.end(), op);
  }

  BrailleCanvas &rect(Color const &color, Rect const &rct,
                      TerminalOp op = TerminalOp::Over) {
    return push()
        .line(color, rct.p1, {rct.p2.x, rct.p1.y}, TerminalOp::Over)
        .line(color, rct.p1, {rct.p1.x, rct.p2.y}, TerminalOp::Over)
        .line(color, rct.p2, {rct.p2.x, rct.p1.y}, TerminalOp::Over)
        .line(color, rct.p2, {rct.p1.x, rct.p2.y}, TerminalOp::Over)
        .pop(op);
  }

  BrailleCanvas &rect(Color const &stroke_color, Color const &fill_color,
                      Rect rct, TerminalOp op = TerminalOp::Over) {
    rct = rct.sorted();
    return push()
        .line(stroke_color, rct.p1, {rct.p2.x, rct.p1.y}, TerminalOp::Over)
        .line(stroke_color, rct.p1, {rct.p1.x, rct.p2.y}, TerminalOp::Over)
        .line(stroke_color, rct.p2, {rct.p2.x, rct.p1.y}, TerminalOp::Over)
        .line(stroke_color, rct.p2, {rct.p1.x, rct.p2.y}, TerminalOp::Over)
        .fill(
            fill_color, rct,
            [r = Rect(rct.p1 + Point(1, 1), rct.p2)](Point p) {
              return r.contains(p);
            },
            TerminalOp::Over)
        .pop(op);
  }

  BrailleCanvas &ellipse(Color const &color, Rect rct,
                         TerminalOp op = TerminalOp::Over) {
    rct = rct.sorted();
    auto size_ = rct.size() + Point(1, 1);

    float x_fac = 2.0f / size_.x;
    Coord y_fac = size_.y / 2 - (!(size_.y % 2)),
          cx = rct.p1.x + (size_.x / cell_cols) - (!(size_.x % cell_cols)),
          cy = rct.p1.y + y_fac;

    return push()
        .stroke(
            color, {rct.p1, {cx, cy}},
            [x_fac, y_fac, cy, x0 = rct.p1.x](Coord x) {
              auto x_over_a = ((x - x0) * x_fac) - 1.0f,
                   next_x_over_a = ((1 + x - x0) * x_fac) - 1.0f;
              Coord base = cy - std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy - std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .stroke(
            color, {{cx + 1, rct.p1.y}, {rct.p2.x, cy}},
            [x_fac, y_fac, cy, x1 = rct.p2.x](Coord x) {
              auto x_over_a = ((x1 - x) * x_fac) - 1.0f,
                   next_x_over_a = ((x1 - x + 1) * x_fac) - 1.0f;
              Coord base = cy - std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy - std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .stroke(
            color, {{rct.p1.x, cy + 1}, {cx, rct.p2.y}},
            [x_fac, y_fac, cy, x0 = rct.p1.x](Coord x) {
              auto x_over_a = ((x - x0) * x_fac) - 1.0f,
                   next_x_over_a = ((1 + x - x0) * x_fac) - 1.0f;
              Coord base = cy + std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy + std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .stroke(
            color, {{cx + 1, cy + 1}, rct.p2},
            [x_fac, y_fac, cy, x1 = rct.p2.x](Coord x) {
              auto x_over_a = ((x1 - x) * x_fac) - 1.0f,
                   next_x_over_a = ((1 + x1 - x) * x_fac) - 1.0f;
              Coord base = cy + std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy + std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .pop(op);
  }

  BrailleCanvas &ellipse(Color const &stroke_color, Color const &fill_color,
                         Rect rct, TerminalOp op = TerminalOp::Over) {
    rct = rct.sorted();
    auto size_ = rct.size() + Point(1, 1);

    float x_fac = 2.0f / size_.x;
    Coord y_fac = size_.y / 2 - (!(size_.y % 2)),
          cx = rct.p1.x + (size_.x / cell_cols) - (!(size_.x % cell_cols)),
          cy = rct.p1.y + y_fac;

    return push()
        .stroke(
            stroke_color, {rct.p1, {cx, cy}},
            [x_fac, y_fac, cy, x0 = rct.p1.x](Coord x) {
              auto x_over_a = ((x - x0) * x_fac) - 1.0f,
                   next_x_over_a = ((1 + x - x0) * x_fac) - 1.0f;
              Coord base = cy - std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy - std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .stroke(
            stroke_color, {{cx + 1, rct.p1.y}, {rct.p2.x, cy}},
            [x_fac, y_fac, cy, x1 = rct.p2.x](Coord x) {
              auto x_over_a = ((x1 - x) * x_fac) - 1.0f,
                   next_x_over_a = ((x1 - x + 1) * x_fac) - 1.0f;
              Coord base = cy - std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy - std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .stroke(
            stroke_color, {{rct.p1.x, cy + 1}, {cx, rct.p2.y}},
            [x_fac, y_fac, cy, x0 = rct.p1.x](Coord x) {
              auto x_over_a = ((x - x0) * x_fac) - 1.0f,
                   next_x_over_a = ((1 + x - x0) * x_fac) - 1.0f;
              Coord base = cy + std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy + std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .stroke(
            stroke_color, {{cx + 1, cy + 1}, rct.p2},
            [x_fac, y_fac, cy, x1 = rct.p2.x](Coord x) {
              auto x_over_a = ((x1 - x) * x_fac) - 1.0f,
                   next_x_over_a = ((1 + x1 - x) * x_fac) - 1.0f;
              Coord base = cy + std::lround(y_fac *
                                            std::sqrt(1 - x_over_a * x_over_a)),
                    end_ = cy + std::lround(y_fac *
                                            std::sqrt(1 - next_x_over_a *
                                                              next_x_over_a));
              return (base != end_) ? std::make_pair(base, end_)
                                    : std::make_pair(base, base + 1);
            },
            TerminalOp::Over)
        .fill(
            fill_color, {rct.p1, {cx, cy}},
            [x_fac, y_fac, cy, x0 = rct.p1.x](Point p) {
              auto x_over_a = ((p.x - x0) * x_fac) - 1.0f;
              Coord base =
                  cy - std::lround(y_fac * std::sqrt(1 - x_over_a * x_over_a));
              return p.y > base;
            },
            TerminalOp::Over)
        .fill(
            fill_color, {{cx + 1, rct.p1.y}, {rct.p2.x, cy}},
            [x_fac, y_fac, cy, x1 = rct.p2.x](Point p) {
              auto x_over_a = ((x1 - p.x) * x_fac) - 1.0f;
              Coord base =
                  cy - std::lround(y_fac * std::sqrt(1 - x_over_a * x_over_a));
              return p.y > base;
            },
            TerminalOp::Over)
        .fill(
            fill_color, {{rct.p1.x, cy + 1}, {cx, rct.p2.y}},
            [x_fac, y_fac, cy, x0 = rct.p1.x](Point p) {
              auto x_over_a = ((p.x - x0) * x_fac) - 1.0f;
              Coord base =
                  cy + std::lround(y_fac * std::sqrt(1 - x_over_a * x_over_a));
              return p.y < base;
            },
            TerminalOp::Over)
        .fill(
            fill_color, {{cx + 1, cy + 1}, rct.p2},
            [x_fac, y_fac, cy, x1 = rct.p2.x](Point p) {
              auto x_over_a = ((x1 - p.x) * x_fac) - 1.0f;
              Coord base =
                  cy + std::lround(y_fac * std::sqrt(1 - x_over_a * x_over_a));
              return p.y < base;
            },
            TerminalOp::Over)
        .pop(op);
  }

  BrailleCanvas &ellipse(Color const &stroke_color, Point const &center,
                         Size const &semiaxes,
                         TerminalOp op = TerminalOp::Over) {
    return ellipse(stroke_color, {center - semiaxes, center + semiaxes}, op);
  }

  BrailleCanvas &ellipse(Color const &stroke_color, Color const &fill_color,
                         Point const &center, Size const &semiaxes,
                         TerminalOp op = TerminalOp::Over) {
    return ellipse(stroke_color, fill_color,
                   {center - semiaxes, center + semiaxes}, op);
  }

private:
  friend value_type;
  template <typename>
  friend std::ostream &
  detail::braille::operator<<(std::ostream &, detail::braille::line_t const &);

  detail::braille::block_t &block(std::size_t ln, std::size_t col) {
    return blocks_[cols_ * ln + col];
  }

  detail::braille::block_t const &block(std::size_t ln, std::size_t col) const {
    return blocks_[cols_ * ln + col];
  }

  detail::braille::block_t &paint(std::size_t ln, std::size_t col,
                                  detail::braille::block_t const &src,
                                  TerminalOp op) {
    auto &dst = block(ln, col);
    return dst = src.paint(dst, op);
  }

  std::size_t lines_ = 0, cols_ = 0;
  detail::braille::image_t blocks_;

  std::forward_list<detail::braille::image_t> stack_;
  std::forward_list<detail::braille::image_t> available_layers_;

  Color background_ = {0, 0, 0, 1};
  TerminalInfo term_;
};

template <typename Fn>
BrailleCanvas &BrailleCanvas::stroke(Color const &color, Rect rct, Fn &&fn,
                                     TerminalOp op) {
  rct = rct.sorted();
  rct.p2 += Point(1, 1);
  rct = rct.clamp(size());
  Rect block_rect{
      {rct.p1.x / cell_cols, rct.p1.y / cell_rows},
      {utils::max(1l, rct.p2.x / cell_cols + (rct.p2.x % cell_cols)),
       utils::max(1l, rct.p2.y / cell_rows + (rct.p2.y % cell_rows != 0))}};

  for (auto ln = block_rect.p1.y; ln < block_rect.p2.y; ++ln) {
    auto line_start = utils::clamp(cell_rows * ln, rct.p1.y, rct.p2.y),
         line_end =
             utils::clamp(cell_rows * ln + cell_rows, rct.p1.y, rct.p2.y);

    for (auto col = block_rect.p1.x; col < block_rect.p2.x; ++col) {
      auto col_start = utils::clamp(cell_cols * col, rct.p1.x, rct.p2.x),
           col_end =
               utils::clamp(cell_cols * col + cell_cols, rct.p1.x, rct.p2.x);

      detail::braille::block_t src(color);

      for (auto x = col_start; x < col_end; ++x) {
        auto ybounds = fn(x);

        if (ybounds.second < ybounds.first)
          ybounds = {ybounds.second + 1, ybounds.first + 1};

        ybounds.first = utils::max(ybounds.first, line_start),
        ybounds.second = utils::min(ybounds.second, line_end);

        for (auto y = ybounds.first; y < ybounds.second; ++y)
          src.set(x, y);
      }

      paint(ln, col, src, op);
    }
  }

  return *this;
}

template <typename Fn>
BrailleCanvas &BrailleCanvas::fill(Color const &color, Rect rct, Fn &&fn,
                                   TerminalOp op) {
  rct = rct.sorted();
  rct.p2 += Point(1, 1);
  rct = rct.clamp(size());
  Rect block_rect{
      {rct.p1.x / cell_cols, rct.p1.y / cell_rows},
      {utils::max(1l, rct.p2.x / cell_cols + (rct.p2.x % cell_cols)),
       utils::max(1l, rct.p2.y / cell_rows + (rct.p2.y % cell_rows != 0))}};

  auto set = [rct, &fn](Point p) { return rct.contains(p) && fn(p); };

  for (auto ln = block_rect.p1.y; ln < block_rect.p2.y; ++ln) {
    auto ybase = cell_rows * ln;
    for (auto col = block_rect.p1.x; col < block_rect.p2.x; ++col) {
      auto xbase = cell_cols * col;
      detail::braille::block_t src(
          color, set({xbase, ybase}), set({xbase, ybase + 1}),
          set({xbase, ybase + 2}), set({xbase, ybase + 3}),
          set({xbase + 1, ybase}), set({xbase + 1, ybase + 1}),
          set({xbase + 1, ybase + 2}), set({xbase + 1, ybase + 3}));

      paint(ln, col, src, op);
    }
  }

  return *this;
}

inline std::ostream &operator<<(std::ostream &stream,
                                BrailleCanvas const &canvas) {
  for (auto const &line : canvas)
    stream << line << '\n';

  return stream;
}

namespace detail {
namespace braille {
inline line_t line_t::next() const {
  return {canvas_, std::next(it_, canvas_->cols_)};
}

template <typename>
std::ostream &operator<<(std::ostream &stream, line_t const &line) {
  auto const &canvas = *line.canvas_;
  auto const &term = canvas.term_;

  // Reset attributes + Bold mode
  // XXX: Empty dots in braille patterns are often rendered as empty
  // XXX: circles unless in bold mode.
  stream << term.reset() << term.bold();

  // Unicode braille patterns are 0x28xx
  // In binary:
  //   0b00101000'xxxxxxxx
  // In UTF-8:
  //   0b1110'0010, 0b10'1000'xx 0b10'xxxxxx

  for (auto it = line.it_, end = line.it_ + canvas.cols_; it != end; ++it) {
    if (it->pixels) {
      stream << term.foreground(
                    it->color.over(canvas.background_).premultiplied())
             << char(0b1110'0010)
             << char(0b10'1000'00 | ((it->pixels & 0b11'000000) >> 6))
             << char(0b10'000000 | (it->pixels & 0b00'111111));
    } else {
      stream << ' ';
    }
  }

  return stream << term.reset();
}
} /* namespace braille */
} /* namespace detail */

} /* namespace plot */

// include/real_canvas.hpp

namespace plot {

template <typename Canvas>
class RealCanvas {
public:
  using coord_type = Coordf;
  using point_type = Pointf;
  using size_type = Sizef;
  using rect_type = Rectf;

  RealCanvas() = default;

  template <typename Arg, typename... Args,
            std::enable_if_t<!std::is_same<std::decay_t<Arg>, Rectf>::value> * =
                nullptr>
  RealCanvas(Arg &&arg, Args &&... args)
      : canvas_(std::forward<Arg>(arg), std::forward<Args>(args)...) {}

  template <typename... Args>
  RealCanvas(Rectf bnds, Args &&... args)
      : bounds_(bnds), canvas_(std::forward<Args>(args)...) {}

  RealCanvas(RealCanvas const &) = default;
  RealCanvas(RealCanvas &&) = default;

  Canvas &canvas() { return canvas_; }

  Canvas const &canvas() const { return canvas_; }

  Rectf bounds() const { return bounds_; }

  void bounds(Rectf bnds) { bounds_ = bnds; }

  Sizef size() const { return bounds_.size(); }

  RealCanvas &push() {
    canvas_.push();
    return *this;
  }

  template <typename... Args>
  RealCanvas &pop(Args &&... args) {
    canvas_.pop(std::forward<Args>(args)...);
    return *this;
  }

  template <typename Size>
  RealCanvas &resize(Size &&sz) {
    canvas_.resize(std::forward<Size>(sz));
    return *this;
  }

  template <typename Size>
  RealCanvas &resize(Rectf bnds, Size &&sz) {
    canvas_.resize(std::forward<Size>(sz));
    bounds_ = bnds;
    return *this;
  }

  RealCanvas &clear() {
    canvas_.clear();
    return *this;
  }

  RealCanvas &clear(Color background) {
    canvas_.clear(background);
    return *this;
  }

  RealCanvas &clear(Rectf rct) {
    canvas_.clear(map(rct));
    return *this;
  }

  template <typename Fn, typename... Args>
  RealCanvas &stroke(Color const &color, Rectf const &rct, Fn &&fn,
                     Args &&... args) {
    canvas_.stroke(
        color, map(rct),
        [this, &fn](typename Canvas::coord_type x) {
          auto real_bounds = fn(unmap(Point(x, 0)).x, unmap(Point(x + 1, 0)).x);
          auto base = map(Pointf(0, real_bounds.first)).y,
               end = map(Pointf(0, real_bounds.second)).y;
          return (base != end) ? std::make_pair(base, end)
                               : std::make_pair(base, base + 1);
        },
        std::forward<Args>(args)...);
    return *this;
  }

  template <typename Fn, typename... Args>
  RealCanvas &fill(Color const &color, Rectf const &rct, Fn &&fn,
                   Args &&... args) {
    canvas_.fill(
        color, map(rct),
        [this, &fn](typename Canvas::point_type p) { return fn(unmap(p)); },
        std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &dot(Color const &color, Pointf p, Args &&... args) {
    canvas_.dot(color, map(p), std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &line(Color const &color, Pointf from, Pointf to,
                   Args &&... args) {
    canvas_.line(color, map(from), map(to), std::forward<Args>(args)...);
    return *this;
  }

  template <typename Iterator, typename... Args>
  RealCanvas &path(Color const &color, Iterator first, Iterator last,
                   Args &&... args) {
    push();
    auto start = *first;
    while (++first != last) {
      auto end = *first;
      line(color, start, end);
      start = end;
    }
    return pop(std::forward<Args>(args)...);
  }

  template <typename... Args>
  RealCanvas &path(Color const &color,
                   std::initializer_list<Pointf> const &points,
                   Args &&... args) {
    return path(color, points.begin(), points.end(),
                std::forward<Args>(args)...);
  }

  template <typename... Args>
  RealCanvas &rect(Color const &color, Rectf const &rct, Args &&... args) {
    canvas_.rect(color, map(rct), std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &rect(Color const &stroke_color, Color const &fill_color,
                   Rectf const &rct, Args &&... args) {
    canvas_.rect(stroke_color, fill_color, map(rct),
                 std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &ellipse(Color const &color, Rectf const &rct, Args &&... args) {
    canvas_.ellipse(color, map(rct), std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &ellipse(Color const &stroke_color, Color const &fill_color,
                      Rectf const &rct, Args &&... args) {
    canvas_.ellipse(stroke_color, fill_color, map(rct),
                    std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &ellipse(Color const &color, Pointf center, Sizef semiaxes,
                      Args &&... args) {
    canvas_.ellipse(color, map(center), map_size(semiaxes),
                    std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  RealCanvas &ellipse(Color const &color, Color const &fill_color,
                      Pointf center, Sizef semiaxes, Args &&... args) {
    canvas_.ellipse(color, fill_color, map(center), map_size(semiaxes),
                    std::forward<Args>(args)...);
    return *this;
  }

  typename Canvas::point_type map(Pointf const &p) const {
    auto canvas_bounds = canvas_.size();
    canvas_bounds -= decltype(canvas_bounds){1, 1};
    return {std::lround((p.x - bounds_.p1.x) / (bounds_.p2.x - bounds_.p1.x) *
                        canvas_bounds.x),
            std::lround((p.y - bounds_.p1.y) / (bounds_.p2.y - bounds_.p1.y) *
                        canvas_bounds.y)};
  }

  typename Canvas::rect_type map(Rectf const &r) const {
    return {map(r.p1), map(r.p2)};
  }

  typename Canvas::size_type map_size(Sizef const &s) const {
    auto sz = this->size();
    auto canvas_bounds = canvas_.size();
    canvas_bounds -= decltype(canvas_bounds){1, 1};
    return {std::lround(s.x / sz.x * canvas_bounds.x),
            std::lround(s.y / sz.y * canvas_bounds.y)};
  }

  Pointf unmap(typename Canvas::point_type const &p) const {
    auto canvas_bounds = canvas_.size();
    canvas_bounds -= decltype(canvas_bounds){1, 1};
    return {(float(p.x) / canvas_bounds.x) * (bounds_.p2.x - bounds_.p1.x) +
                bounds_.p1.x,
            (float(p.y) / canvas_bounds.y) * (bounds_.p2.y - bounds_.p1.y) +
                bounds_.p1.y};
  }

  Rectf unmap(typename Canvas::rect_type const &r) const {
    return {unmap(r.p1), unmap(r.p2)};
  }

  Sizef unmap_size(typename Canvas::size_type const &s) const {
    auto sz = this->size();
    auto canvas_bounds = canvas_.size();
    canvas_bounds -= decltype(canvas_bounds){1, 1};
    return {float(s.x) / canvas_bounds.x * sz.x,
            float(s.y) / canvas_bounds.y * sz.y};
  }

private:
  Rectf bounds_{{0.0f, 1.0f}, {1.0f, 0.0f}};
  Canvas canvas_;
};

template <typename Canvas>
inline std::ostream &operator<<(std::ostream &stream,
                                RealCanvas<Canvas> const &canvas) {
  return stream << canvas.canvas();
}

namespace detail {
// Make RealCanvas a valid block
template <typename Canvas, bool IsCanvas>
struct block_ref_traits<plot::RealCanvas<Canvas>, IsCanvas> {
  using iterator = typename Canvas::const_iterator;

  static Size size(plot::RealCanvas<Canvas> const &block) {
    return block.canvas().char_size();
  }

  static iterator begin(plot::RealCanvas<Canvas> const &block) {
    return block.canvas().begin();
  }

  static iterator end(plot::RealCanvas<Canvas> const &block) {
    return block.canvas().end();
  }
};
} /* namespace detail */

} /* namespace plot */

// include/plot.hpp
