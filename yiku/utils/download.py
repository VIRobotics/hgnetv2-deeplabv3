import requests
import re,os
from urllib.parse import urlparse
import pathlib
from pip._vendor.rich.progress import (
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    Progress
)
class  IntegrityError(Exception):
    pass
def download_from_url(url,dir_path):
    a = urlparse(url)
    fname = os.path.basename(a.path)
    if os.path.isfile(os.path.join(dir_path,fname)):
        return os.path.join(dir_path,fname)
    response=requests.get(url,stream=True,allow_redirects=True)
    if "Content-Disposition" in response.headers.keys():
        d=response.headers["Content-Disposition"]
        pathlib.Path(os.path.join(dir_path,fname)).touch()
        fname = re.findall("filename=(.+)", d)[0]
    print("下载文件:%s"%fname)
    path=os.path.join(dir_path,fname)
    block_size = 1024  # 1 Kibibyte
    if 'Content-Encoding' in response.headers.keys():
        unlimited=True
    else:
        unlimited = False


    if unlimited:
        progress = Progress(
            SpinnerColumn(),
            "{task.description}",
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeElapsedColumn()
        )
        task1 = progress.add_task("[red]Downloading %s" % fname,total=None)
        progress.start()
        with open(path, mode="wb") as f:
            for data in response.iter_content(block_size):
                progress.update(task1,advance=len(data))
                f.write(data)

        progress.stop()
        return

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    # progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    progress = Progress(
        SpinnerColumn(),
        "{task.description}",
        BarColumn(),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task1 = progress.add_task("[red]Downloading %s"%fname, total=total_size_in_bytes)
    l=0
    progress.start()
    with open(path,mode="wb")as f:
        for data in response.iter_content(block_size):
            progress.update(task1,advance=len(data))
            f.write(data)
            l=l+len(data)
    progress.stop()
    if total_size_in_bytes != 0 and l != total_size_in_bytes:
        os.remove(path)
        raise IntegrityError

    return path


if __name__=="__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        download_from_url("https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",d)