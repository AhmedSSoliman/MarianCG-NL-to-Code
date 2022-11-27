# MarianCG: A Code Generation Transformer Model Inspired by Machine Translation

```
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mariancg-a-code-generation-transformer-model/code-generation-on-conala)](https://paperswithcode.com/sota/code-generation-on-conala?p=mariancg-a-code-generation-transformer-model)
```


In this work we worked to improve the solving of the code generation problem and implement a transformer model that can work with high accurate results. We implemented MarianCG transformer model which is a code generation model that can be able to generate code from natural language. This work declares the impact of using Marian machine translation model for solving the problem of code generation. In our implementation we prove that a machine translation model can be operated and working as a code generation model.Finally, we set the new contributors and state-of-the-art on CoNaLa reaching a BLEU score of 34.43 in the code generation problem with CoNaLa dataset.

This is the model is available on the huggingface hub
https://huggingface.co/AhmedSSoliman/MarianCG-CoNaLa


CoNaLa Dataset for Code Generation is available at
https://huggingface.co/datasets/AhmedSSoliman/CoNaLa


This model is available in spaces using gridio at: https://huggingface.co/spaces/AhmedSSoliman/MarianCG-CoNaLa

```python
# Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "AhmedSSoliman/MarianCG-NL-to-Code"
model = AutoModelForSeq2SeqLM.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa")
tokenizer = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa")

# Input (Natural Language) and Output (Python Code)
NL_input = "create array containing the maximum value of respective elements of array `[2, 3, 4]` and array `[1, 5, 2]"
output = model.generate(**tokenizer(NL_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt"))
output_code = tokenizer.decode(output[0], skip_special_tokens=True)



```

# Citation

We now have a paper for this work and you can cite:

```
@article{soliman2022mariancg,
  title={MarianCG: a code generation transformer model inspired by machine translation},
  author={Soliman, Ahmed S and Hadhoud, Mayada M and Shaheen, Samir I},
  journal={Journal of Engineering and Applied Science},
  volume={69},
  number={1},
  pages={1--23},
  year={2022},
  publisher={SpringerOpen}
}

```
