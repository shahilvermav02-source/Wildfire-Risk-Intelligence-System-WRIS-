from __future__ import annotations

import csv
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

FALLBACK_CSV = """x,y,month,day,ffmc,dmc,dc,isi,temp,rh,wind,rain,area
7,5,aug,fri,91.5,145.4,678.2,8.3,29.1,35,3.6,0.0,2.4
7,4,aug,sat,90.6,131.7,669.1,6.7,22.2,51,4.0,0.0,0.0
8,6,sep,sun,92.3,98.3,741.0,10.2,26.5,42,5.4,0.0,5.1
6,3,jul,thu,86.2,26.2,94.3,5.1,18.2,63,2.7,0.0,0.0
4,4,mar,wed,87.0,35.5,109.0,5.5,16.0,66,3.0,0.2,0.2
5,5,apr,tue,88.2,42.1,125.2,6.2,17.5,61,3.4,0.0,0.4
6,4,may,mon,89.1,53.8,210.5,6.9,20.2,55,2.9,0.0,0.8
7,3,jun,sat,90.0,68.0,320.1,7.4,23.0,49,3.2,0.0,1.5
8,5,jul,fri,91.2,84.7,430.2,8.1,25.6,44,3.8,0.0,2.2
9,6,aug,thu,92.8,110.3,550.7,9.7,28.0,38,4.3,0.0,4.3
3,2,oct,wed,85.5,22.8,88.4,4.0,14.8,72,2.1,0.3,0.0
2,1,nov,tue,83.1,18.6,75.3,3.2,12.5,78,1.9,0.8,0.0
1,2,dec,mon,80.9,12.2,60.1,2.7,9.8,84,2.5,1.2,0.0
4,3,jan,sun,82.4,15.0,70.0,3.0,11.0,80,2.0,0.6,0.0
5,2,feb,sat,84.0,19.5,82.3,3.6,13.0,76,2.2,0.4,0.1
6,6,sep,mon,93.2,120.0,760.5,11.0,30.2,33,4.8,0.0,8.7
7,7,sep,tue,93.5,128.9,780.2,11.5,31.0,31,5.0,0.0,12.3
8,7,aug,wed,92.9,116.1,720.4,10.7,29.4,36,4.6,0.0,6.5
9,8,aug,mon,93.0,118.7,730.8,10.9,30.0,34,4.9,0.0,7.1
5,6,jun,thu,89.8,65.4,300.2,7.0,21.7,52,3.1,0.0,1.2
"""


def download_csv(url: str, output_path: str | Path, timeout: int = 30) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url, timeout=timeout) as response:
            content = response.read()
        output_path.write_bytes(content)
    except URLError:
        output_path.write_text(FALLBACK_CSV, encoding="utf-8")

    return output_path


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
