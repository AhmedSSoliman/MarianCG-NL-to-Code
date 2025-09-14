# MarianCG: Code Generation Transformer Model Inspired by Machine Translation [![License: MIT][License-Badge]](LICENSE)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mariancg-a-code-generation-transformer-model/code-generation-on-conala)](https://paperswithcode.com/sota/code-generation-on-conala?p=mariancg-a-code-generation-transformer-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mariancg-a-code-generation-transformer-model/code-generation-on-django)](https://paperswithcode.com/sota/code-generation-on-django?p=mariancg-a-code-generation-transformer-model)

## Introduction

**MarianCG** is a transformer-based model for code generation, inspired by advances in machine translation. It leverages the Marian machine translation architecture to translate natural language (NL) instructions into executable code, achieving state-of-the-art results on standard code generation benchmarks. MarianCG is designed for researchers and developers who want to automate code synthesis from natural language descriptions.

- **CoNaLa (Code/Natural Language Challenge):** BLEU score of 34.43 (SOTA)
- **DJANGO dataset:** Exact match accuracy of 81.83%

This repository provides code, pretrained models, and evaluation scripts for reproducing our results and applying MarianCG to your own code generation tasks.

---

## Table of Contents
- [Model: CoNaLa Dataset](#model-conala-dataset)
- [Model: DJANGO Dataset](#model-django-dataset)
- [Usage](#usage)
- [Example Output](#example-output)
- [Installation & Requirements](#installation--requirements)
- [Datasets](#datasets)
- [Citation](#citation)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

---

## Model: CoNaLa Dataset

- **Model on Hugging Face Hub:** [MarianCG-CoNaLa-Large](https://huggingface.co/AhmedSSoliman/MarianCG-CoNaLa-Large)
- **Demo Space:** [Hugging Face Spaces](https://huggingface.co/spaces/AhmedSSoliman/MarianCG-CoNaLa-Large)
- **Implementation Notebook:** [MarianCG_CoNaLa_Large.ipynb](https://github.com/AhmedSSoliman/MarianCG-NL-to-Code/blob/master/Experiments/MarianCG-CoNaLa-Large/MarianCG_CoNaLa_Large.ipynb) &nbsp;&nbsp; [![Open in Colab][Colab Badge]][RDP Notebook]

#### Quick Start Example
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large")
tokenizer = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large")

NL_input = "create array containing the maximum value of respective elements of array `[2, 3, 4]` and array `[1, 5, 2]"
inputs = tokenizer(NL_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
output = model.generate(**inputs)
output_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_code)
```

---

## Model: DJANGO Dataset

- **Model on Hugging Face Hub:** [MarianCG-DJANGO](https://huggingface.co/AhmedSSoliman/MarianCG-DJANGO)
- **Demo Space:** [Hugging Face Spaces](https://huggingface.co/spaces/AhmedSSoliman/MarianCG-DJANGO)
- **Implementation Notebook:** [MarianCG_DJANGO.ipynb](https://github.com/AhmedSSoliman/MarianCG-NL-to-Code/blob/master/Experiments/MarianCG-DJANGO/MarianCG_DJANGO.ipynb) &nbsp;&nbsp; [![Open in Colab][Colab Badge]][Code Notebook]

#### Quick Start Example
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("AhmedSSoliman/MarianCG-DJANGO")
tokenizer = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG-DJANGO")

NL_input = "define the method i with an argument self."
inputs = tokenizer(NL_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
output = model.generate(**inputs)
output_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_code)
```

---

## Example Output

**Input (Natural Language):**
```
create array containing the maximum value of respective elements of array [2, 3, 4] and array [1, 5, 2]
```
**Output (Python Code):**
```
[max(a, b) for a, b in zip([2, 3, 4], [1, 5, 2])]
```

---

## Installation & Requirements

- Python 3.7+
- [transformers](https://huggingface.co/docs/transformers/index)
- [torch](https://pytorch.org/)

Install dependencies:
```bash
pip install transformers torch
```

---

## Usage

1. Load the model and tokenizer as shown in the examples above.
2. Provide a natural language instruction as input and generate code.

---

## Datasets

- **CoNaLa-Large:** [Hugging Face Dataset](https://huggingface.co/datasets/AhmedSSoliman/CoNaLa-Large)
- **DJANGO:** [Hugging Face Dataset](https://huggingface.co/datasets/AhmedSSoliman/DJANGO)

---

## Citation

If you use MarianCG in your research, please cite our paper:

```
@article{soliman2022mariancg,
  title={MarianCG: a code generation transformer model inspired by machine translation},
  author={Soliman, Ahmed S and Hadhoud, Mayada M and Shaheen, Samir I},
  journal={Journal of Engineering and Applied Science},
  volume={69},
  number={1},
  pages={1--23},
  year={2022},
  publisher={SpringerOpen},
  url={https://doi.org/10.1186/s44147-022-00159-4}
}
```

---

## Contributing

Contributions are welcome! Please open an issue or pull request for suggestions, bug reports, or improvements. For major changes, please open an issue first to discuss what you would like to change.

---

## Support

If you find this project useful, please consider:
1. **Starring this repository**
2. **Promoting this repository**
3. **Contributing to this repository**

For questions or discussions, please open an issue on GitHub.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

[Colab Badge]:          https://colab.research.google.com/assets/colab-badge.svg
[License-Badge]:        https://img.shields.io/badge/License-MIT-blue.svg
[RDP Issues]:           https://img.shields.io/github/issues/PradyumnaKrishna/Colab-Hacks/Colab%20RDP?label=Issues
[RDP Notebook]:         https://colab.research.google.com/drive/1HtGfWOwBx0deii0WPQD3o_NfGRYBEs1w?usp=sharing
[Code Issues]:          https://img.shields.io/github/issues/PradyumnaKrishna/Colab-Hacks/Code%20Server?label=Issues
[Code Notebook]:        https://colab.research.google.com/drive/1Hcj3akrYFe3bKHNCj-g1qrRHSbLdkk9s?usp=sharing
