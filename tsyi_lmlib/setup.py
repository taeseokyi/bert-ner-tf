from gettext import install
from setuptools import setup

setup(
    name='tsyi_lmlib',
    version='0.0.0',
    description='tsyi pip install test',
    url='https://github.com/taeseokyi/bert-ner-tf.git',
    author='taeseok yi',
    author_email='tsyi@kisti.re.kr',
    license='taeseok yi',
    packages=['tsyi_lmlib'],
    zip_safe=False,
    install_requires=[
        'numpy==1.19.2'
    ]
)