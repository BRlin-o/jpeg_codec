# JPEG_codec
## map
```
jpeg_codec/
│
├── encoder.py            # 包含Encoder類
├── decoder.py            # 包含Decoder類
├── utils/                # 通用工具和函數
│   ├── __init__.py
│   ├── color_transforms.py
│   ├── huffman.py
│   ├── quantization.py
│   └── ...               # 其他工具
├── tests/                # 單元測試
│   ├── __init__.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── ...               # 其他測試文件
│
├── examples/             # 範例代碼
│   ├── encode_image.py
│   └── decode_image.py
│
├── setup.py              # 安裝和分發包所需的設置
├── README.md             # 介紹您的庫的信息
└── LICENSE               # 版權和許可信息
```