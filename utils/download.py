import requests
import tqdm
import re,os
from urllib.parse import urlparse
import pathlib
class  IntegrityError(Exception):
    pass
def download_from_url(url,dir_path):
    a = urlparse(url)
    fname = os.path.basename(a.path)
    if os.path.isfile(os.path.join(dir_path,fname)):
        return
    response=requests.get(url,stream=True,allow_redirects=True)
    if "Content-Disposition" in response.headers.keys():
        d=response.headers["Content-Disposition"]
        pathlib.Path(os.path.join(dir_path,fname)).touch()
        fname = re.findall("filename=(.+)", d)[0]
    path=os.path.join(dir_path,fname)
    block_size = 1024  # 1 Kibibyte
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(path,mode="wb")as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        os.remove(path)
        raise IntegrityError
