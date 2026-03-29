# Run WRIS in Google Colab (Single Cell)

Copy and run this cell in Google Colab:

```python
!git clone -b codex/build-wildfire-risk-intelligence-system https://github.com/shahilvermav02-source/Wildfire-Risk-Intelligence-System-WRIS-.git wris
%cd /content/wris

import os
os.environ['PYTHONPATH'] = '/content/wris/src'

!python scripts/download_data.py
!python scripts/train_model.py

!nohup python scripts/run_api.py > /tmp/wris_api.log 2>&1 &
!sleep 3

!curl -s http://127.0.0.1:8000/health
!curl -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"x":7.4,"y":4.1,"month":"aug","day":"fri","ffmc":91.5,"dmc":145.4,"dc":678.2,"isi":8.3,"temp":29.1,"rh":35,"wind":3.6,"rain":0.0}'
```

GUI in Colab runtime: `http://127.0.0.1:8000/gui`.
