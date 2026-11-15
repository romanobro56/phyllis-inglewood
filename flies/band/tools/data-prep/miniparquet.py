"""Tiny flat-schema Parquet reader (test harness only; the real script uses pandas+pyarrow).

Supports: PLAIN / dictionary encodings, data page v1/v2, OPTIONAL/REQUIRED columns,
codecs UNCOMPRESSED, SNAPPY (pure python), BROTLI (via node's zlib).
"""
import struct
import subprocess
import sys

import numpy as np


# ---------------- thrift compact protocol ----------------
class TReader:
    def __init__(self, buf, pos=0):
        self.b = buf
        self.p = pos

    def byte(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self):
        shift = 0
        result = 0
        while True:
            c = self.byte()
            result |= (c & 0x7F) << shift
            if not c & 0x80:
                return result
            shift += 7

    def zz(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def read_value(self, t):
        if t == 1:
            return True
        if t == 2:
            return False
        if t == 3:
            v = self.byte()
            return v - 256 if v > 127 else v
        if t in (4, 5, 6):
            return self.zz()
        if t == 7:
            v = struct.unpack_from('<d', self.b, self.p)[0]
            self.p += 8
            return v
        if t == 8:
            n = self.varint()
            v = bytes(self.b[self.p:self.p + n])
            self.p += n
            return v
        if t in (9, 10):
            h = self.byte()
            size = h >> 4
            et = h & 0x0F
            if size == 15:
                size = self.varint()
            out = []
            for _ in range(size):
                if et in (1, 2):
                    out.append(self.byte() == 1)
                else:
                    out.append(self.read_value(et))
            return out
        if t == 11:
            size = self.varint()
            out = {}
            if size:
                kv = self.byte()
                for _ in range(size):
                    k = self.read_value(kv >> 4)
                    out[k] = self.read_value(kv & 0x0F)
            return out
        if t == 12:
            return self.struct()
        raise ValueError(f'unknown thrift type {t}')

    def struct(self):
        out = {}
        fid = 0
        while True:
            h = self.byte()
            if h == 0:
                return out
            t = h & 0x0F
            d = h >> 4
            fid = fid + d if d else self.zz()
            out[fid] = self.read_value(t)


# ---------------- codecs ----------------
def snappy_decompress(src):
    src = memoryview(src)
    r = TReader(src)
    n = r.varint()
    out = bytearray(n)
    o = 0
    p = r.p
    L = len(src)
    while p < L:
        tag = src[p]
        p += 1
        kind = tag & 3
        if kind == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(src[p:p + nb], 'little')
                p += nb
            ln += 1
            out[o:o + ln] = src[p:p + ln]
            p += ln
            o += ln
            continue
        if kind == 1:
            ln = 4 + ((tag >> 2) & 7)
            off = ((tag >> 5) << 8) | src[p]
            p += 1
        elif kind == 2:
            ln = (tag >> 2) + 1
            off = src[p] | (src[p + 1] << 8)
            p += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(src[p:p + 4], 'little')
            p += 4
        start = o - off
        if off >= ln:
            out[o:o + ln] = out[start:start + ln]
        else:  # overlapping copy: repeat the pattern
            pat = bytes(out[start:o])
            reps = (ln // off) + 1
            out[o:o + ln] = (pat * reps)[:ln]
        o += ln
    return bytes(out)


def brotli_decompress(src):
    js = ("const z=require('zlib');const c=[];process.stdin.on('data',d=>c.push(d));"
          "process.stdin.on('end',()=>process.stdout.write(z.brotliDecompressSync(Buffer.concat(c))));")
    return subprocess.run(['node', '-e', js], input=bytes(src), capture_output=True, check=True).stdout


def decompress(codec, data):
    if codec == 0:
        return bytes(data)
    if codec == 1:
        return snappy_decompress(data)
    if codec == 4:
        return brotli_decompress(data)
    raise NotImplementedError(f'codec {codec}')


# ---------------- encodings ----------------
def rle_hybrid(buf, pos, end, bit_width, count):
    """Decode RLE/bit-packed hybrid into an int array of `count` values."""
    out = np.empty(count, dtype=np.int64)
    k = 0
    r = TReader(buf, pos)
    bw_bytes = (bit_width + 7) // 8
    while k < count and r.p < end:
        h = r.varint()
        if h & 1:
            groups = h >> 1
            nvals = groups * 8
            nbytes = groups * bit_width
            raw = np.frombuffer(bytes(buf[r.p:r.p + nbytes]), dtype=np.uint8)
            r.p += nbytes
            if bit_width == 0:
                vals = np.zeros(nvals, dtype=np.int64)
            else:
                bits = np.unpackbits(raw, bitorder='little')
                bits = bits[:nvals * bit_width].reshape(nvals, bit_width).astype(np.int64)
                vals = (bits << np.arange(bit_width, dtype=np.int64)).sum(axis=1)
            take = min(nvals, count - k)
            out[k:k + take] = vals[:take]
            k += take
        else:
            run = h >> 1
            v = int.from_bytes(bytes(buf[r.p:r.p + bw_bytes]), 'little') if bw_bytes else 0
            r.p += bw_bytes
            take = min(run, count - k)
            out[k:k + take] = v
            k += take
    return out[:k]


NP_TYPES = {1: np.int32, 2: np.int64, 4: np.float32, 5: np.float64}


def plain_values(ptype, data, pos, n):
    if ptype in NP_TYPES:
        dt = np.dtype(NP_TYPES[ptype]).newbyteorder('<')
        return np.frombuffer(data, dtype=dt, count=n, offset=pos).copy(), pos + n * dt.itemsize
    if ptype == 6:
        out = []
        for _ in range(n):
            ln = struct.unpack_from('<i', data, pos)[0]
            pos += 4
            out.append(bytes(data[pos:pos + ln]).decode())
            pos += ln
        return np.array(out, dtype=object), pos
    raise NotImplementedError(f'physical type {ptype}')


# ---------------- reader ----------------
def read_parquet(path, columns=None):
    with open(path, 'rb') as f:
        buf = f.read()
    flen = struct.unpack('<i', buf[-8:-4])[0]
    meta = TReader(buf, len(buf) - 8 - flen).struct()
    schema = meta[2]
    leaves = {}
    for el in schema[1:]:
        leaves[el[4].decode()] = el.get(3, 0)  # repetition: 0 required, 1 optional
    result = {}
    for rg in meta[4]:
        for cc in rg[1]:
            cm = cc[3]
            name = b'.'.join(cm[3]).decode()
            if columns is not None and name not in columns:
                continue
            ptype, codec, nvals = cm[1], cm[4], cm[5]
            max_def = 1 if leaves.get(name, 0) == 1 else 0
            start = cm.get(11) or cm[9]
            if cm.get(11) is not None:
                start = min(cm[11], cm[9])
            end = start + cm[7]
            pos = start
            dictionary = None
            parts = []
            seen = 0
            while pos < end and seen < nvals:
                r = TReader(buf, pos)
                ph = r.struct()
                pos = r.p
                csize = ph[3]
                raw = memoryview(buf)[pos:pos + csize]
                pos += csize
                ptype_page = ph[1]
                if ptype_page == 2:  # dictionary page
                    data = decompress(codec, raw)
                    dictionary, _ = plain_values(ptype, data, 0, ph[7][1])
                    continue
                if ptype_page == 0:
                    dph = ph[5]
                    n = dph[1]
                    enc = dph[2]
                    data = decompress(codec, raw)
                    dpos = 0
                    if max_def:
                        ln = struct.unpack_from('<i', data, 0)[0]
                        defs = rle_hybrid(data, 4, 4 + ln, 1, n)
                        dpos = 4 + ln
                        nn = int(defs.sum())
                    else:
                        defs = None
                        nn = n
                elif ptype_page == 3:
                    dph = ph[8]
                    n = dph[1]
                    enc = dph[4]
                    dl, rl = dph[5], dph[6]
                    lv = bytes(raw[:rl + dl])
                    body = raw[rl + dl:]
                    data = lv + (decompress(codec, body) if dph.get(7, True) else bytes(body))
                    if max_def and dl:
                        defs = rle_hybrid(data, rl, rl + dl, 1, n)
                        nn = int(defs.sum())
                    else:
                        defs, nn = None, n - dph.get(2, 0)
                    dpos = rl + dl
                else:
                    continue
                if enc == 0:
                    vals, _ = plain_values(ptype, data, dpos, nn)
                elif enc in (2, 8):
                    bw = data[dpos]
                    idx = rle_hybrid(data, dpos + 1, len(data), bw, nn)
                    vals = dictionary[idx]
                else:
                    raise NotImplementedError(f'encoding {enc}')
                if defs is not None and nn != n:
                    raise NotImplementedError('nulls present')
                parts.append(vals)
                seen += n
            result.setdefault(name, []).append(np.concatenate(parts) if parts else np.array([]))
    return {k: np.concatenate(v) for k, v in result.items()}


def describe(path):
    with open(path, 'rb') as f:
        buf = f.read()
    flen = struct.unpack('<i', buf[-8:-4])[0]
    meta = TReader(buf, len(buf) - 8 - flen).struct()
    print(path, 'rows', meta[3], 'row groups', len(meta[4]))
    for cc in meta[4][0][1]:
        cm = cc[3]
        print('  ', b'.'.join(cm[3]).decode(), 'type', cm[1], 'codec', cm[4], 'enc', cm[2])


if __name__ == '__main__':
    for p in sys.argv[1:]:
        describe(p)
