from setuptools import setup

setup(name="motion_tracker",
      version='0.1',
      description='tools for recording, tracking and anayzing video, \
      particularly of animals in behavioral experiments.',
      url="https://github.com/jpcurrea/FIU-DPB/motion_tracker.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      packages=['motion_tracker'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'scikit-image',
          'scikit-learn',
          'scikit-video',
          'PyQt5'
      ],
      zip_safe=False)
