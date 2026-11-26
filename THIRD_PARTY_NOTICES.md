# Third-party notices

VGEO is licensed under the GNU General Public License, version 3 or (at your
option) any later version (see `LICENSE`). It includes or builds against the
following software, under their own licenses, all compatible with the GPL.

| Component | Used in | License |
|---|---|---|
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | compiled into `vgeo_stream` (the add-on's native library); `web/meshopt_decoder.mjs` | MIT |
| [GLFW](https://github.com/glfw/glfw) | the original Vulkan proof of concept (viewer), fetched at build time | zlib |
| [pybind11](https://github.com/pybind/pybind11) | the original Vulkan proof of concept (`vgeo_native` module), fetched at build time | BSD 3-Clause |

## meshoptimizer

MIT License

Copyright (c) 2016-2026 Arseny Kapoulkine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
