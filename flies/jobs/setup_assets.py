"""Fetch pinned local station dependency. Run explicitly; never downloads in step()."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

SHA256='ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb'
URL='https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js'


def main():
    target=Path(__file__).with_name('vendor')/'three.module.js'
    target.parent.mkdir(exist_ok=True)
    data=target.read_bytes() if target.exists() else urlopen(URL,timeout=60).read()
    if hashlib.sha256(data).hexdigest()!=SHA256:
        raise ValueError('Three.js dependency checksum mismatch')
    target.write_bytes(data)
    print('Pinned local Three.js asset ready:',target)


if __name__=='__main__':main()
