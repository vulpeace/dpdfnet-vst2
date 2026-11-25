#include "rfft.h"

#include <cmath>

namespace sherpa_onnx {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline RfftComplex Add(RfftComplex a, RfftComplex b) {
  return {a.re + b.re, a.im + b.im};
}

inline RfftComplex Sub(RfftComplex a, RfftComplex b) {
  return {a.re - b.re, a.im - b.im};
}

inline RfftComplex Mul(RfftComplex a, RfftComplex b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

void Factor(int32_t n, std::vector<int32_t> *out) {
  int32_t p = 4;
  auto floor_sqrt = static_cast<int32_t>(std::sqrt(static_cast<double>(n)));
  do {
    while (n % p) {
      switch (p) {
        case 4:
          p = 2;
          break;
        case 2:
          p = 3;
          break;
        default:
          p += 2;
          break;
      }
      if (p > floor_sqrt) p = n;
    }
    n /= p;
    out->push_back(p);
    out->push_back(n);
  } while (n > 1);
}
}

Rfft::Rfft(int32_t n) : n_(n), m_(n / 2) {
  Factor(m_, &factors_);

  twiddles_.resize(m_);
  for (int32_t j = 0; j != m_; ++j) {
    double phase = -2.0 * kPi * j / m_;
    twiddles_[j] = {static_cast<float>(std::cos(phase)),
                    static_cast<float>(std::sin(phase))};
  }

  half_twiddles_.resize(m_ + 1);
  for (int32_t k = 0; k <= m_; ++k) {
    double phase = -kPi * k / m_;
    half_twiddles_[k] = {static_cast<float>(std::cos(phase)),
                         static_cast<float>(std::sin(phase))};
  }

  packed_.resize(m_);
  spectrum_.resize(m_);

  int32_t max_radix = 0;
  for (size_t i = 0; i < factors_.size(); i += 2) {
    if (factors_[i] > max_radix) max_radix = factors_[i];
  }
  bfly_.resize(max_radix);
}

void Rfft::Forward(const float *in, float *out) const {
  for (int32_t j = 0; j != m_; ++j) {
    packed_[j] = {in[2 * j], in[2 * j + 1]};
  }
  CpxFft(packed_.data(), spectrum_.data(), false);

  for (int32_t k = 0; k != m_; ++k) {
    RfftComplex a = spectrum_[k];
    RfftComplex b = spectrum_[(m_ - k) % m_];
    b.im = -b.im;

    RfftComplex fe = {0.5f * (a.re + b.re), 0.5f * (a.im + b.im)};
    RfftComplex diff = {a.re - b.re, a.im - b.im};
    RfftComplex fo = {0.5f * diff.im, -0.5f * diff.re};

    RfftComplex x = Add(fe, Mul(half_twiddles_[k], fo));
    out[2 * k] = x.re;
    out[2 * k + 1] = x.im;
  }

  out[2 * m_] = spectrum_[0].re - spectrum_[0].im;
  out[2 * m_ + 1] = 0.0f;
}

void Rfft::Inverse(const float *in, float *out) const {
  for (int32_t k = 0; k != m_; ++k) {
    RfftComplex x = (k == 0) ? RfftComplex{in[0], 0.0f}
                             : RfftComplex{in[2 * k], in[2 * k + 1]};
    RfftComplex y = (k == 0) ? RfftComplex{in[2 * m_], 0.0f}
                             : RfftComplex{in[2 * (m_ - k)],
                                           -in[2 * (m_ - k) + 1]};

    RfftComplex fe = {0.5f * (x.re + y.re), 0.5f * (x.im + y.im)};
    RfftComplex diff = {0.5f * (x.re - y.re), 0.5f * (x.im - y.im)};
    RfftComplex tw = half_twiddles_[k];
    tw.im = -tw.im;
    RfftComplex fo = Mul(diff, tw);

    packed_[k] = {fe.re - fo.im, fe.im + fo.re};
  }

  CpxFft(packed_.data(), spectrum_.data(), true);

  const float scale = 1.0f / static_cast<float>(m_);
  for (int32_t j = 0; j != m_; ++j) {
    out[2 * j] = spectrum_[j].re * scale;
    out[2 * j + 1] = spectrum_[j].im * scale;
  }
}

void Rfft::CpxFft(const RfftComplex *in, RfftComplex *out,
                  bool inverse) const {
  Work(out, in, 1, 1, factors_.data(), inverse);
}

void Rfft::Work(RfftComplex *out, const RfftComplex *in, int32_t fstride,
                int32_t in_stride, const int32_t *factors,
                bool inverse) const {
  RfftComplex *out_begin = out;
  const int32_t p = factors[0];
  const int32_t m = factors[1];
  RfftComplex *out_end = out + p * m;

  if (m == 1) {
    do {
      *out = *in;
      in += fstride * in_stride;
    } while (++out != out_end);
  } else {
    do {
      Work(out, in, fstride * p, in_stride, factors + 2, inverse);
      in += fstride * in_stride;
    } while ((out += m) != out_end);
  }

  out = out_begin;
  switch (p) {
    case 2:
      Butterfly2(out, fstride, m, inverse);
      break;
    case 4:
      Butterfly4(out, fstride, m, inverse);
      break;
    default:
      ButterflyGeneric(out, fstride, m, p, inverse);
      break;
  }
}

void Rfft::Butterfly2(RfftComplex *out, int32_t fstride, int32_t m,
                      bool inverse) const {
  for (int32_t i = 0; i != m; ++i) {
    RfftComplex tw = twiddles_[fstride * i];
    if (inverse) tw.im = -tw.im;
    RfftComplex t = Mul(out[i + m], tw);
    out[i + m] = Sub(out[i], t);
    out[i] = Add(out[i], t);
  }
}

void Rfft::Butterfly4(RfftComplex *out, int32_t fstride, int32_t m,
                      bool inverse) const {
  const int32_t m2 = 2 * m;
  const int32_t m3 = 3 * m;
  for (int32_t i = 0; i != m; ++i) {
    RfftComplex tw1 = twiddles_[fstride * i];
    RfftComplex tw2 = twiddles_[fstride * 2 * i];
    RfftComplex tw3 = twiddles_[fstride * 3 * i];
    if (inverse) {
      tw1.im = -tw1.im;
      tw2.im = -tw2.im;
      tw3.im = -tw3.im;
    }

    RfftComplex s0 = Mul(out[i + m], tw1);
    RfftComplex s1 = Mul(out[i + m2], tw2);
    RfftComplex s2 = Mul(out[i + m3], tw3);

    RfftComplex s5 = Sub(out[i], s1);
    RfftComplex f0 = Add(out[i], s1);
    RfftComplex s3 = Add(s0, s2);
    RfftComplex s4 = Sub(s0, s2);

    out[i + m2] = Sub(f0, s3);
    out[i] = Add(f0, s3);

    if (inverse) {
      out[i + m] = {s5.re - s4.im, s5.im + s4.re};
      out[i + m3] = {s5.re + s4.im, s5.im - s4.re};
    } else {
      out[i + m] = {s5.re + s4.im, s5.im - s4.re};
      out[i + m3] = {s5.re - s4.im, s5.im + s4.re};
    }
  }
}

void Rfft::ButterflyGeneric(RfftComplex *out, int32_t fstride, int32_t m,
                            int32_t p, bool inverse) const {
  RfftComplex *scratch = bfly_.data();
  for (int32_t u = 0; u != m; ++u) {
    int32_t k = u;
    for (int32_t q1 = 0; q1 != p; ++q1) {
      scratch[q1] = out[k];
      k += m;
    }

    k = u;
    for (int32_t q1 = 0; q1 != p; ++q1) {
      int32_t twidx = 0;
      out[k] = scratch[0];
      for (int32_t q = 1; q != p; ++q) {
        twidx += fstride * k;
        if (twidx >= m_) twidx -= m_;
        RfftComplex tw = twiddles_[twidx];
        if (inverse) tw.im = -tw.im;
        out[k] = Add(out[k], Mul(scratch[q], tw));
      }
      k += m;
    }
  }
}
}
