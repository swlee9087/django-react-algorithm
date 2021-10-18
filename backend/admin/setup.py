from setuptools import setup, find_packages

setup_requires = [
]

install_requires = [
    # 'django==3.1.8'
]

dependency_links = [
    'git+https://github.com/django/django.git@stable/1.6.x#egg=Django-1.6b4',
]

# setup(
#     name='Root App',
#     version='0.1',
#     description='Root App',
#     author='tsumomo',
#     author_email='sw_lee9087@naver.com',
#     packages=find_packages(),
#     install_requires=install_requires,
#     setup_requires=setup_requires,
#     dependency_links=dependency_links,
#     scripts=['../manage.py'],
#     entry_points={
#         'console_scripts': [
#             'publish = admin.sorting.script:main',
#         ],
#     },
# )
setup(name='sortingExample',

      version='1.2.0.0',

      description='Extension Tools for sortingExample',

      author='swlee9087',

      author_email='sw_lee9087@naver.com',

      license='MIT',

      py_modules=['calWordsFreq', 'findKeywordSentences'],

      python_requires='>=3',

      packages=['sorting']

      )
