from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'gitpython==3.1.32',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch==1.4.0',
        'torchvision==0.5.0',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
