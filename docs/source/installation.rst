Installation
============


Using pip
---------

.. attention::

   SysSimPyMMEN will be pip installable soon!


From source
-----------

All of the code is publicly available on `Github <https://github.com/hematthi/SysSimPyMMEN>`_. Thus an alternative to pip is to download or clone the repository and install it from there:

.. code-block:: bash

   git clone https://github.com/hematthi/SysSimPyMMEN.git
   python -m pip install .

There are multiple ways of cloning; if you have trouble with the above or prefer another, see `this Github guide <https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories>`_. You should also fork the repository first if you want to make your own changes to the source code later.


Installing SysSimPyPlots
------------------------

In order to use SysSimPyMMEN, you will also need to install `SysSimPyPlots <https://syssimpyplots.readthedocs.io/>`_ -- see the `installation page <https://syssimpyplots.readthedocs.io/en/latest/installation.html>`_ for that package.

That page will also include instructions for downloading some simulated catalogs. These are not required for using all of the functions in SysSimPyMMEN, but would be required for applying the functions to those simulated systems and running some of the examples in the following tutorials.

You are now ready to use SysSimPyMMEN!


Dependencies
------------

SysSimPyMMEN has been tested on Python >3.7 and uses:

- ``numpy`` (for almost everything)
- ``matplotlib`` (for making plots)
- ``scipy`` (for some miscellaneous functions)
- ``syssimpyplots`` (for some core functions and loading catalogs, as described above)
