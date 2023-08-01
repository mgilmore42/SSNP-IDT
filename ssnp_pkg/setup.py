from setuptools import setup

setup(
    entry_points={
        'console_scripts': [
            'ssnp = ssnp.__main__:main',
        ],
    }
)
