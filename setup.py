from setuptools import setup, find_packages

setup(name="cmap_factory",
      version="0.0.1",
      author="Tao Jing",
      author_email="jingt20@mails.tsinghua.edu.cn",
      description="Generate color maps from images, color palettes, or other sources.",
      packages=find_packages(),
      install_requires=[
          'networkx', 'numpy', 'matplotlib', 'scipy', 'scikit-image',
          'scikit-learn'
      ])
