Hi Rossella,

Many thanks for speaking to Florian. Pip is much appreciated.

I’m still having issues using python3 as pip is bound to python2 so I’m unable to install the correct packages.

Specifically, if I use ```pip install numpy``` then the module successfully installs but is only available to python2. However if I use ``pip3 install numpy``, I get an import error stating ``ImportError: cannot import name 'sysconfig'``. I get the same error when I run ``python3 -m pip install numpy``.


From what I can gather, this is an issue with the disutils module and may be solved by the root command ```sudo apt install python3-distutils```. If Florian agrees that this is the issue, I would massively appreciate him running this.
