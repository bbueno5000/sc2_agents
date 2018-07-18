# MIT License
#
# Copyright (c) 2018 Benjamin Bueno (bbueno5000)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup


setup(name='StarCraft Agents',
      version='0.1',
      description='Starcraft II agents',
      author='bbueno5000',
      author_email='bbueno5000@gmail.com',
      license='MIT License',
      keywords='StarCraft AI',
      url='https://github.com/bbueno5000/starcraft_agents',
      packages=['starcraft_agents',
                'starcraft_agents.bin',
                'starcraft_agents.deepq',
                'starcraft_agents.minigame_agents'],
      install_requires=['pysc2', 'tensorflow==1.4'],
      entry_points={'console_scripts': ['dqn_agent = starcraft_agents.bin.dqn_agent:entry_point']})
