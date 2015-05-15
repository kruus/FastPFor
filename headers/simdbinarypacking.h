/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire, http://lemire.me/en/
 */

#ifndef SIMDBINARYPACKING_H_
#define SIMDBINARYPACKING_H_

#include "codecs.h"
#include "simdbitpacking.h"
#include "util.h"

namespace FastPForLib {

/**
 *
 * Designed by D. Lemire with ideas from Leonid Boystov. This scheme is NOT patented.
 *
 * Code data in miniblocks of 128 integers.
 * To maintain 16-byte write alignment in fastpack, we group
 * 16 such miniblocks into a block of 16 * 128 = 2048 integers.
 *
 * Reference and documentation:
 *
 * Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second through vectorization
 * http://arxiv.org/abs/1209.2137
 */
class SIMDBinaryPacking: public IntegerCODEC {
public:
    static const uint32_t CookiePadder = 123456;
    static const uint32_t MiniBlockSize = 128;
    static const uint32_t HowManyMiniBlocks = 16;
    static const uint32_t BlockSize = MiniBlockSize;

    /**
     * The way this code is written, it will automatically "pad" the
     * header according to the alignment of the out pointer. So if you
     * move the data around, you should preserve the alignment.
     */
    void encodeArray(const uint32_t *in, const size_t length, uint32_t *out,
            size_t &nvalue) {
        checkifdivisibleby(length, BlockSize);
        const uint32_t * const initout(out);
        *out++ = static_cast<uint32_t>(length);
        while(needPaddingTo128Bits(out)) *out++ = CookiePadder;
        uint32_t Bs[HowManyMiniBlocks];
#if defined(EJK)
        // [ejk] loop merge and lightly optimize --> ~ 7-15% speedup on codecs tests 1,2,3
        uint32_t const* inNxt = in + BlockSize;
        for (const uint32_t * const final = in + length; inNxt <= final; inNxt += BlockSize) {
            uint32_t *out0 = out;
            out += HowManyMiniBlocks / 4U;      // save space for Bs[16] info
            uint32_t const* mbNxt = in;
            for (uint32_t i=0U; i < HowManyMiniBlocks;  in = mbNxt, ++i ) {
                mbNxt += MiniBlockSize;
                Bs[i] = maxbits(in, mbNxt);
                SIMD_fastpackwithoutmask_32(in, reinterpret_cast<__m128i *>(out), Bs[i]);
                out += MiniBlockSize/32U * Bs[i];
#else
        const uint32_t *const final = in + length;
        for (; in + HowManyMiniBlocks * MiniBlockSize
                 <= final; in += HowManyMiniBlocks * MiniBlockSize) {

            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i)
                Bs[i] = maxbits(in + i * MiniBlockSize,
                        in + (i + 1) * MiniBlockSize);
            *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8)
                | Bs[3];
            *out++ = (Bs[4] << 24) | (Bs[5] << 16) | (Bs[6] << 8)
                            | Bs[7];
            *out++ = (Bs[8] << 24) | (Bs[9] << 16) | (Bs[10] << 8)
                            | Bs[11];
            *out++ = (Bs[12] << 24) | (Bs[13] << 16) | (Bs[14] << 8)
                            | Bs[15];
            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
                SIMD_fastpackwithoutmask_32(in + i * MiniBlockSize, reinterpret_cast<__m128i *>(out),
                                Bs[i]);
                out += MiniBlockSize/32 * Bs[i];
#endif
            }
#if defined(EJK)
            assert( in = inNxt ); // after HowManyMiniBlocks, 'in' has advanced by exactly BlockSize
            static_assert( HowManyMiniBlocks == 16, "hard-wired for HowManyMiniBlocks == 16");
            *out0 = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3];
            *++out0 = (Bs[4] << 24) | (Bs[5] << 16) | (Bs[6] << 8) | Bs[7];
            *++out0 = (Bs[8] << 24) | (Bs[9] << 16) | (Bs[10] << 8) | Bs[11];
            *++out0 = (Bs[12] << 24) | (Bs[13] << 16) | (Bs[14] << 8) | Bs[15];
#endif
        }
        if (in < final) {
            const size_t howmany = (final - in) / MiniBlockSize;
            memset(&Bs[0], 0, HowManyMiniBlocks * sizeof(uint32_t));
            for (uint32_t i = 0; i < howmany; ++i)
                Bs[i] = maxbits(in + i * MiniBlockSize,
                        in + (i + 1) * MiniBlockSize);
            *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8)
                     | Bs[3];
            *out++ = (Bs[4] << 24) | (Bs[5] << 16) | (Bs[6] << 8)
                     | Bs[7];
            *out++ = (Bs[8] << 24) | (Bs[9] << 16) | (Bs[10] << 8)
                     | Bs[11];
            *out++ = (Bs[12] << 24) | (Bs[13] << 16) | (Bs[14] << 8)
                     | Bs[15];
            for (uint32_t i = 0; i < howmany; ++i) {
                SIMD_fastpackwithoutmask_32(in + i * MiniBlockSize, reinterpret_cast<__m128i *>(out),
                                Bs[i]);
                out += MiniBlockSize/32 * Bs[i];
            }
            in += howmany * MiniBlockSize;
            assert(in == final);
        }

        nvalue = out - initout;
    }

    const uint32_t * decodeArray(const uint32_t *in, const size_t /*length*/,
            uint32_t *out, size_t & nvalue) {
        const uint32_t actuallength = *in++;
        if(needPaddingTo128Bits(out)) throw std::runtime_error("bad initial output align");
        while(needPaddingTo128Bits(in)) {
            if(in[0] != CookiePadder) throw std::logic_error("SIMDBinaryPacking alignment issue.");
            ++in;
        }
        const uint32_t * const initout(out);
        uint32_t Bs[HowManyMiniBlocks];
        for (; out < initout + actuallength / (HowManyMiniBlocks * MiniBlockSize) *HowManyMiniBlocks * MiniBlockSize ;
             out += HowManyMiniBlocks * MiniBlockSize) {
            for(uint32_t i = 0; i < 4 ; ++i,++in) {
                Bs[0 + 4 * i] = static_cast<uint8_t>(in[0] >> 24);
                Bs[1 + 4 * i] = static_cast<uint8_t>(in[0] >> 16);
                Bs[2 + 4 * i] = static_cast<uint8_t>(in[0] >> 8);
                Bs[3 + 4 * i] = static_cast<uint8_t>(in[0]);
            }
            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
                // D.L. : is the reinterpret_cast safe here?
                SIMD_fastunpack_32(reinterpret_cast<const __m128i *>(in), out + i * MiniBlockSize, Bs[i]);
                in += MiniBlockSize/32 * Bs[i];
            }
        }
        if (out < initout + actuallength) {
            const size_t howmany = (initout + actuallength - out) / MiniBlockSize;
            for (uint32_t i = 0; i < 4 ; ++i, ++in) {
                Bs[0 + 4 * i] = static_cast<uint8_t>(in[0] >> 24);
                Bs[1 + 4 * i] = static_cast<uint8_t>(in[0] >> 16);
                Bs[2 + 4 * i] = static_cast<uint8_t>(in[0] >> 8);
                Bs[3 + 4 * i] = static_cast<uint8_t>(in[0]);
            }
            for (uint32_t i = 0; i < howmany; ++i) {
                SIMD_fastunpack_32(reinterpret_cast<const __m128i *>(in), out + i * MiniBlockSize, Bs[i]);
                         in += MiniBlockSize/32 * Bs[i];
            }
            out += howmany * MiniBlockSize;
            assert(out ==  initout + actuallength);
        }
        nvalue = out - initout;
        return in;
    }

    std::string name() const {
        return "SIMDBinaryPacking";
    }

};


} // namespace FastPFor

#endif /* SIMDBINARYPACKING_H_ */
