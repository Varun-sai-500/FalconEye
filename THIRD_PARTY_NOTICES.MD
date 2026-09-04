# Third-Party Notices

This project incorporates or depends on third-party models and source code. The following notices are provided in accordance with the licensing terms of those components.

---

## 1. Segment Anything Model (SAM)

- **Component:** `facebook/sam-vit-base`
- **Source:** [https://huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
- **Provider:** Meta AI (FAIR)
- **Usage in this project:** Loaded via the Hugging Face `transformers` library for segmentation.
- **License:** Apache License 2.0
- **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.

> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
>
> http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## 2. CLIPSeg

- **Component:** CLIPSeg (CLIP-based segmentation model)
- **Source:** [https://huggingface.co/CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- **Provider:** CIDAS, University of Göttingen
- **Usage in this project:** Loaded via the Hugging Face `transformers` library for open-vocabulary/text-prompted segmentation.
- **License:** Apache License 2.0
- **Copyright:** Copyright (c) the CLIPSeg authors.

> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
>
> http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## 3. DaSiamRPN Tracker

- **Component:** DaSiamRPN visual object tracker, source code adapted for this project
- **Original repository:** [https://github.com/foolwood/DaSiamRPN](https://github.com/foolwood/DaSiamRPN)
- **Copyright:** Copyright (c) 2018, Qiang Wang
- **License:** MIT License
- **Weights:** Redistributed via this project's Hugging Face Hub repository, with original license and provenance stated there.
- **Modifications:** The implementation in this repository has been adapted from the upstream source, integrated into this project's GPU-accelerated tracking pipeline.

> MIT License
>
> Copyright (c) 2018 Qiang Wang
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

