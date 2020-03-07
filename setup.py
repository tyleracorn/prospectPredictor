from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='prospectpredictor',
      version='0.1.1',
      description='prospectivity prediction based on GIS shapefiles',
      long_description=readme(),
      long_description_content_type='text/x-rst',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
      ],
      keywords='GIS prospectivity varriogram shapefiles raster',
      url='https://github.com/tyleracorn/prospectPredictor',
      author='Tyler Acorn',
      author_email='tyler.acorn@gmail.com',
      license='MIT',
      packages=['prospectpredictor'],
      install_requires=[
          'geopandas',
          'rasterio',
          'descartes',
          'matplotlib',
      ],
      include_package_data=True,
     zip_safe=False)