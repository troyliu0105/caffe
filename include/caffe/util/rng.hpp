#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <algorithm>
#include <iterator>
#include <random>

//#include "boost/random/mersenne_twister.hpp"
//#include "boost/random/uniform_int.hpp"

#include "caffe/common.hpp"

namespace caffe {

typedef std::mt19937 rng_t;

inline rng_t *caffe_rng() {
  return static_cast<caffe::rng_t *>(Caffe::rng_stream().generator());
}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                    RandomGenerator *gen) {
  std::shuffle(begin, end, *gen);
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
  shuffle(begin, end, caffe_rng());
}
} // namespace caffe

#endif // CAFFE_RNG_HPP_
