# Nhận dạng ký tự biển số xe (OCR) 

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Cấu trúc thư mục dự án

```text
.
|-- LICENSE                 # Giay phep su dung ma nguon
|-- Makefile                #
|-- checkpoint/             # Checkpoint
|   |-- best.pt.tar         # Checkpoint tốt nhất
|   |-- last.pt.tar         
|   |-- no_ctc_v2best.pt.tar
|   |-- no_ctc_v2last.pt.tar
|   |-- ocr_no_ctcbest.pt.tar
|   `-- ocrlast.pt.tar
|-- config/
|   `-- test.yaml           # Cấu hình cho pipeline test
|-- models/
|   `-- ocr.py              # Where model OCR is defined
|-- ocr_test.py             # Script test
|-- ocr_train.py            # Script train
|-- requirements.txt        # Dependency
|-- result.txt              
`-- utils.py                
```


## Cài đặt

Yêu cầu:

- Python 3.9+
- pip

Cài môi trường và dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Chạy train:

```bash
python ocr_train.py --batch-size 16 --lr 1e-4 --epochs 200
```

Chạy test:

```bash
python ocr_test.py --batch-size 16
```

Hoặc dùng Makefile:

```bash
make train
make test
```



