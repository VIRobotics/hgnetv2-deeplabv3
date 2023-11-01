from pathlib import Path

from setuptools import setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        (List[str]): List of parsed requirements.
    """

    requirements = []
    for line in Path(file_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line.split('#')[0].strip())  # ignore inline comments

    return requirements



setup(
    name='yiku_seg',  # name of pypi package
    version="0.1.1",  # version of pypi package
    python_requires='>=3.8',
    license='AGPL-3.0',
    description=('SegCollection'),
    url='hhttps://gitee.com/yiku-ai/hgnetv2-deeplabv3',
    author='香草琪猫猫',
    author_email='',
    packages=['yiku'] + [str(x) for x in Path('yiku').rglob('*/') if x.is_dir() and '__' not in str(x)],
    package_data={
        '': ['*.pth'],
        'ultralytics.assets': ['*.jpg',"*.pth"]},
    include_package_data=True,
    install_requires=parse_requirements(PARENT / 'requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows', ],
    keywords='machine-learning, deep-learning, vision, ML, DL, AI, ImageSegment ,DeepLab, Transformer',
    entry_points={'console_scripts': ['siren.train = yiku.train:main', 'siren.export= yiku.export:main']})